package classification

import DataLoader
import aucRoc

fun main() {
    val MAX_DEPTH = 9
    val MIN_NODES = 4

    val allData = DataLoader("data/comp2_train.csv").read()
    val (target, data) = allData.splitColsBy(1)

    val model1 =  DecisionTreeTuner(0.7, MAX_DEPTH - 2, MIN_NODES + 1, (0 until 11).toList(), DecisionTreeClassifier.Uncertanty.SQUARED).getBest(10, data, target)
    val model2 =  DecisionTreeTuner(0.7, MAX_DEPTH, MIN_NODES, (0 until 11 union (12..12)).toList(), DecisionTreeClassifier.Uncertanty.G05).getBest(10, data, target)
    val model3 =  DecisionTreeTuner(0.7, MAX_DEPTH - 2, MIN_NODES + 1, (0 until 11).toList(), DecisionTreeClassifier.Uncertanty.SQUARED).getBest(10, data, target)
    val model4 =  DecisionTreeTuner(0.7, MAX_DEPTH, MIN_NODES, (0 until 11 union (12..12)).toList(), DecisionTreeClassifier.Uncertanty.G05).getBest(10, data, target)
    val model5 =  DecisionTreeTuner(0.7, MAX_DEPTH, MIN_NODES, (0 until 11 union (11..11)).toList(), DecisionTreeClassifier.Uncertanty.G15).getBest(10, data, target)
    val model6 =  DecisionTreeTuner(0.5, MAX_DEPTH, MIN_NODES, (0 until 11).toList(), DecisionTreeClassifier.Uncertanty.GINI).getBest(15, data, target)
    val model7 =  DecisionTreeTuner(1.0, MAX_DEPTH + 1, MIN_NODES - 1, (0 until 12).toList(), DecisionTreeClassifier.Uncertanty.SQUARED).getBest(7, data, target)
    val model8 =  DecisionTreeTuner(1.0, MAX_DEPTH - 2, MIN_NODES - 1, (0 until 11 union (11..11)).toList(), DecisionTreeClassifier.Uncertanty.ENT).getBest(7, data, target)
    val model9 =  DecisionTreeTuner(1.0, MAX_DEPTH, MIN_NODES, (0 until 11 union (11..11)).toList(), DecisionTreeClassifier.Uncertanty.GINI).getBest(7, data, target)
    val model10 =  DecisionTreeTuner(1.0, MAX_DEPTH - 1, MIN_NODES, (0 until 12).toList(), DecisionTreeClassifier.Uncertanty.SQUARED).getBest(7, data, target)
    val model11 =  DecisionTreeTuner(1.0, MAX_DEPTH, MIN_NODES + 1, (0 until 12).toList(), DecisionTreeClassifier.Uncertanty.ENT).getBest(7, data, target)

    val model = RandomForest(model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11)

    val cv = AggregatingCrossValidation(model, ::aucRoc)
    cv.setK(20)

    println(cv.getQuality(data, target))

    val unknown = DataLoader("data/comp2_test.csv").read()

    val res = model.predict(unknown)

    DataLoader("res.csv").writeCsvIndexed(res, listOf("target", "Id"))
}