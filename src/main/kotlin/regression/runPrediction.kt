package regression

import DataLoader
import meanSqError

fun main() {
    /*
    val MAX_DEPTH = 5
    val MIN_LEAFS = 3
    val model = regression.DecisionTreeRegressor(MAX_DEPTH, MIN_LEAFS)
     */

    val SEED = 58L
    val INIT_STEP = 7.0
    val STEP_POW = 1.95
    val LAMBD = 0.4
    val ALG_EPS = 0.01
    val GRAD_RED = 0.095
    val model = LinearRegression(INIT_STEP, STEP_POW, LAMBD, ALG_EPS, GRAD_RED)
    //model.setSeed(SEED)


    val alldata = DataLoader("data/comp1_train.csv").read()
    val (target, data) = alldata.splitColsBy(1)

    model.fit(data, target)

    val unknown = DataLoader("data/comp1_test.csv").read()

    val res = model.predict(unknown)

    println(meanSqError(res, target))

    DataLoader("res.csv").writeCsvIndexed(res, listOf("target", "Id"))
}