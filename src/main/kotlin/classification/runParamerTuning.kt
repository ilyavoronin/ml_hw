package classification

import DataLoader
import ParamVars
import ParameterTuner
import aucRoc
import regression.LinearRegression

fun main() {
    val cv = AggregatingCrossValidation(DecisionTreeClassifier(), ::aucRoc)
    cv.setK(10)

    val alldata = DataLoader("data/comp2_train.csv").read().dropNan()

    val (target, data) = alldata.splitColsBy(1)

    val tuner = ParameterTuner(cv, data, target, 3)

    val bestParams = tuner.findBest(
            false,
            ParamVars("maxDepth", 6.0, 7.0, 8.0, 9.0),
            ParamVars("minNodeData", 4.0, 5.0, 6.0)
    )

    bestParams.print()
}