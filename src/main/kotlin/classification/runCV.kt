package classification

import DataLoader
import aucRoc

fun main() {
    val MAX_DEPTH = 9
    val MIN_NODES = 4

    val model1 =  DecisionTreeClassifier(MAX_DEPTH - 1, MIN_NODES, (0 until 11).toList(), DecisionTreeClassifier.Uncertanty.GINI) //RandomForest(TREES_CNT, MAX_DEPTH, MIN_NODES, CNT_PARAMS)
    val model2 =  DecisionTreeClassifier(MAX_DEPTH, MIN_NODES, (0 until 11).toList(), DecisionTreeClassifier.Uncertanty.ENT)
    val model3 =  DecisionTreeClassifier(MAX_DEPTH - 2, MIN_NODES + 1, (0 until 11).toList(), DecisionTreeClassifier.Uncertanty.SQUARED)
    val model4 =  DecisionTreeClassifier(MAX_DEPTH, MIN_NODES, (0 until 11 union (12..12)).toList(), DecisionTreeClassifier.Uncertanty.G05)
    val model5 =  DecisionTreeClassifier(MAX_DEPTH, MIN_NODES, (0 until 11 union (11..11)).toList(), DecisionTreeClassifier.Uncertanty.G15)

    val model = RandomForest(model1, model2, model3, model4, model5)


    val allData = DataLoader("data/comp2_train.csv").read()
    val (target, data) = allData.splitColsBy(1)

    val cv = AggregatingCrossValidation(model, ::aucRoc)
    cv.setK(20)

    println(cv.getQuality(data, target))
}