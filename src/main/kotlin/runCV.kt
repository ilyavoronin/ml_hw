

fun main() {
    /*
    val MAX_DEPTH = 5
    val MIN_LEAFS = 3
    val model = DecisionTreeRegressor(MAX_DEPTH, MIN_LEAFS)

     */

    val SEED = 58L
    val INIT_STEP = 7.0
    val STEP_POW = 1.95
    val LAMBD = 0.4
    val ALG_EPS = 0.01
    val GRAD_RED = 0.095
    val model = LinearRegression(INIT_STEP, STEP_POW, LAMBD, ALG_EPS, GRAD_RED)
    model.setSeed(SEED)

    val alldata = DataLoader("data/comp1_train.csv").read()

    val (target, data) = alldata.splitColsBy(1)

    val cv = CrossValidation(model).withK(100)
    cv.setSeed(SEED)

    val cvRes = cv.getQuality(data, target)
    println(cvRes)
    println(cvRes.mean())
    println(cvRes.max())
}