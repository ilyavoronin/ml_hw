package classification

import DataFrame
import Params

interface ClassificationModel {
    fun setParams(params: Params)

    fun fit(table: DataFrame, target: DataFrame)

    fun predict(table: DataFrame): DataFrame
}