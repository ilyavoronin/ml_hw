

fun main() {
    val MAX_DEPTH = 5
    val MIN_LEAFS = 3

    val alldata = DataLoader("comp1_train.csv").read()

    val (target, data) = alldata.splitColsBy(1)

    val model = DecisionTreeRegressor(MAX_DEPTH, MIN_LEAFS)

    val cvRes = CrossValidation().withK(10).getQuality(model, data, target)
    println(cvRes)
    println(cvRes.mean())
    println(cvRes.max())
}