package classification

import DataLoader

fun main() {
    val MAX_DEPTH = 9
    val MIN_NODES = 4

    val model1 =  DecisionTreeClassifier(MAX_DEPTH - 1, MIN_NODES, (0 until 11).toList(), DecisionTreeClassifier.Uncertanty.GINI) //RandomForest(TREES_CNT, MAX_DEPTH, MIN_NODES, CNT_PARAMS)
    val model2 =  DecisionTreeClassifier(MAX_DEPTH, MIN_NODES, (0 until 11).toList(), DecisionTreeClassifier.Uncertanty.ENT)
    val model3 =  DecisionTreeClassifier(MAX_DEPTH - 2, MIN_NODES + 1, (0 until 11).toList(), DecisionTreeClassifier.Uncertanty.SQUARED)
    val model4 =  DecisionTreeClassifier(MAX_DEPTH, MIN_NODES, (0 until 11 union (12..12)).toList(), DecisionTreeClassifier.Uncertanty.G05)
    val model5 =  DecisionTreeClassifier(MAX_DEPTH, MIN_NODES, (0 until 11 union (11..11)).toList(), DecisionTreeClassifier.Uncertanty.G15)

    val model = RandomForest(model1, model2, model3, model4, model5)

    val alldata = DataLoader("data/comp2_train.csv").read()

    val (target, data) = alldata.splitColsBy(1)

    model.fit(data, target)

    val unknown = DataLoader("data/comp2_test.csv").read()

    val res = model.predict(unknown)

    DataLoader("res.csv").writeCsvIndexed(res, listOf("target", "Id"))
}