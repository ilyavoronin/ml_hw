

interface RegressionModel {
    fun setParams(params: Params)

    fun fit(table: DataFrame, target: DataFrame)

    fun predict(table: DataFrame): DataFrame
}