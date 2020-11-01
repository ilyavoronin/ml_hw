

interface RegressionModel {
    fun fit(table: DataFrame, target: DataFrame)

    fun predict(table: DataFrame): DataFrame
}