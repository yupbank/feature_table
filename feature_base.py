from operator import itemgetter, or_, and_
from functools import reduce
import copy


from pyspark.sql.types import StringType, ArrayType
import pyspark.sql.functions as F


class Lit(object):

    def __init__(self, value):
        self.value = value


class FeatureColumn(object):
    """
    A FeaturesColumn object is a stateless transformation.
    """

    def __init__(self, 
                 default_value=None,
                 input_columns=None,
                 initial_values=None):
        self.default_value = default_value
        self.input_columns = input_columns
        self.initial_values = initial_values

    def __call__(self, **kwargs):
        new_obj = copy.deepcopy(self)
        new_obj.__dict__.update(kwargs)
        return new_obj

    @property
    def default(self):
        return self.default_value

    @property
    def inputs(self):
        return self.input_columns

    def process_df(self, df, output, env=None):
        raise NotImplementedError

    def get_input_column(self, output):
        raise NotImplementedError


class Identity(FeatureColumn):

    def get_input_column(self, output):
        if self.inputs:
            return self.inputs[0]
        else:
            return output

    def process_df(self, df, output, env=None):
        assert output == self.get_input_column(output)
        return df


def _column_names_to_expression(names, df=None):
    if df is None:
        get_col = F.col
    else:
        names = filter(lambda r: r in df.columns, names)
        get_col = df.__getitem__

    col_expression = []
    for name in names:
        if isinstance(name, Lit):
            col = F.lit(name.value)
        elif isinstance(name, basestring):
            if '.' in name:
                col = None
                for w in name.split('.'):
                    if col is None:
                        col = get_col(w)
                    else:
                        col = col[w]
            else:
                col = get_col(name)
        else:
            col = name

        col_expression.append(col)

    return col_expression


def _column_equal(df_one, df_two, keys_one, keys_two):
    if not isinstance(keys_one, (list, tuple)):
        keys_one = [keys_one]
    if not isinstance(keys_two, (list, tuple)):
        keys_two = [keys_two]
    return reduce(and_, ((col_one == col_two)
                         for col_one, col_two in
                         zip(_column_names_to_expression(keys_one, df_one),
                             _column_names_to_expression(keys_two, df_two))))


class OneToOne(Identity):

    def _process_df(self, col):
        return col

    def process_df(self, df, output, env=None):
        col_name = self.get_input_column(output)
        col = _column_names_to_expression([col_name])[0]
        return df.withColumn(output, self._process_df(col))


class Concat(OneToOne):

    def process_df(self, df, output, env=None):
        col_exprs = _column_names_to_expression(self.input_columns)
        return df.withColumn(output, F.concat(*col_exprs))


class IsIn(OneToOne):

    def _process_df(self, col):
        return col.isin(self.initial_values)


class Contains(OneToOne):

    def _process_df(self, col):
        not_in_condition = lambda _: (col.like('%' + _ + '%'))
        return reduce(or_, map(not_in_condition, self.initial_values))


class NotContains(Contains):

    def _process_df(self, col):
        return ~super(NotContains, self)._process_df(col)


class IsNull(OneToOne):

    def _process_df(self, col):
        return col.isNull()


class IsNotNull(OneToOne):

    def _process_df(self, col):
        return col.isNotNull()


class Unicode_(OneToOne):

    def _process_df(self, col):
        return col.cast('string')


class MapUnicode(OneToOne):

    def _process_df(self, col):
        return col.cast(ArrayType(StringType()))


class Float_(OneToOne):

    def _process_df(self, col):
        return F.when(col.cast('float')).isNull(), F.lit(self.default_value)).otherwise(
            col.cast('float'))


class Log1p(OneToOne):

    def _process_df(self, col):
        return F.when(col.isNull(), F.lit(self.default_value)).otherwise(
            F.log1p(col.cast('float')))


class Combinator(object):

    def __init__(self, feature_columns):
        self.feature_columns = feature_columns

    @property
    def default(self):
        return self.feature_columns[-1][1].default

    def process_df(self, df, output, env=None):
        for feature, feature_column in self.feature_columns:
            df = feature_column.process_df(df, feature)
        return df


class Joiner(object):

    def __init__(self, key, with_key, feature_table=None, with_feature_table=None, default_value=None, how='left'):
        self.key = key
        self.with_key = with_key
        self.with_feature_table = with_feature_table
        self.default_value = default_value
        self.feature_table = feature_table
        self.how = how
        assert how in ('left', 'inner'), 'only left and inner is supported'

    def join_df(self, df, table, with_key):
        left, right = df, table

        if self.feature_table:
            if self.how == 'left':
                left, right = table, df

        return (
            left
            .join(right, on=with_key, how=self.how)
        )

    def prepare_table(self, env, output):
        if self.with_feature_table:
            origin_key = self.with_key
            join_key = self.key
            processor = self.with_feature_table
        else:
            origin_key = self.key
            join_key = self.with_key
            processor = self.feature_table

        table = processor.squash_df(env=env, origin_key=origin_key, output=output)
        return table.withColumnRenamed(origin_key, join_key), join_key

    def process_df(self, df, output, env=None):
        table, join_key = self.prepare_table(env, output)
        return self.join_df(df, table, join_key)

    def get_input_column(self, output):
        raise Exception('Too complicated to count inputs for a joiner')


class MapJoiner(Joiner):

    def prepare_df(self, df, explode_key):
        t = df.select(self.key).schema.fields[0].dataType.elementType
        exploded_df = df.withColumn(explode_key,
                                    F.explode(
                                        F.when(F.col(self.key).isNotNull(),
                                               F.col(self.key))
                                        .otherwise(F.array(F.lit(None).cast(t)))))
        return exploded_df

    def process_df(self, df, output, env=None):
        assert self.with_feature_table, 'only join against is supported for map join'
        explode_key = 'explode:%s' % self.key
        assert explode_key not in df.columns, 'duplication happened'
        table, join_key = self.prepare_table(env, output)

        table = table.withColumnRenamed(self.key, explode_key)

        exploded_df = self.prepare_df(df, explode_key)

        new_df = self.join_df(exploded_df, table, explode_key)

        new_df = (
            new_df
            .withColumn(output, F.when(F.col(explode_key).isNotNull(), F.create_map(explode_key, output))
                        .otherwise(F.lit(None)))
            .groupby(_column_names_to_expression(df.columns, exploded_df))
            .agg(F.collect_list(F.col(output)).alias(output))
        )
        return new_df


class FeatureTable(object):

    def __init__(self, feature_columns, name=None):
        self.feature_columns = feature_columns
        self.name = name

    @property
    def outputs(self):
        return map(itemgetter(0), self.feature_columns)

    def process_df(self, df=None, env=None):
        if env:
            df = env.get(self.name)

        for feature, feature_column in self.feature_columns:
            df = feature_column.process_df(df, feature, env)
        return df

    def squash_df(self, df=None, env=None, origin_key=None, output=None):
        df = self.process_df(df, env)
        df = df.withColumn(output, F.struct(*outputs))
        return df.select(origin_key, output)

    def transform_df(self, df, env=None):
        return df.select(*self.outputs)


class Executor(object):

    def __init__(self, feature_table):
        if not isinstance(feature_table, FeatureTable):
            self.feature_table = FeatureTable(feature_table)
        else:
            self.feature_table = feature_table

    def process_df(self, df=None, env=None):
        return self.feature_table.process_df(df, env)

    def transform_df(self, df=None, env=None):
        return self.feature_table.transform_df(df, env)
