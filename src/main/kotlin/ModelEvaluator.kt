interface ModelEvaluator<T> {
    fun getQuality(model: RegressionModel, data: DataFrame, target: DataFrame): T
}