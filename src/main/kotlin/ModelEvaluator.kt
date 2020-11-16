interface ModelEvaluator<T> {
    val model: MLModel

    fun getQuality(data: DataFrame, target: DataFrame): T

    fun getDoubleQuality(data: DataFrame, target: DataFrame): Double

    fun setModelParams(params: Params) {
        model.setParams(params)
    }
}