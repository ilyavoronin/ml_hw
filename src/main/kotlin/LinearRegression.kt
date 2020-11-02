import java.lang.IllegalStateException
import java.util.*
import kotlin.math.abs
import kotlin.math.pow

class LinearRegression(
    var initialStepSize: Double,
    var stepPow: Double,
    var lam: Double,
    private var algConvEps: Double,
    private var gradRed: Double
) : RegressionModel {
    constructor(): this(1.0, 1.9, 0.1, 0.01, 0.1)

    private val derConvEps: Double = 1e-9
    private var myRandom = Random()

    private var myWeights: List<Double>? = null

    fun setSeed(seed: Long) {
        myRandom = Random(seed)
    }

    override fun setParams(params: Params) {
        params.getParam("initialStepSize")?.let { initialStepSize = it }
        params.getParam("lam")?.let { lam = it }
        params.getParam("algConvEps")?.let { algConvEps = it }
        params.getParam("gradRed")?.let { gradRed = it }
        params.getParam("stepPow")?.let { stepPow = it }
    }

    override fun fit(table: DataFrame, target: DataFrame) {
        var weights = initWeights(table)
        val gradients = table.getRows().mapIndexed {i, row -> calcGrad(createFuncForPoint(row, target[i][0]), weights) }.toMutableList()
        var festimation = 0.0
        table.getRows().forEachIndexed {i, row ->
            festimation += calcErr(row, target[i][0], weights)
        }
        festimation /= table.rowsCnt()

        var lastEst = festimation + 1e9
        var sumGrad = gradients.reduce {a, b -> a.plus(b)}

        var bestRes = 1e9

        var stepNum = 1

        while (abs(lastEst - festimation) > algConvEps) {
            val i = myRandom.nextInt(table.rowsCnt())
            val row = table[i]
            val ans = target[i][0]

            val gradStepSize = initialStepSize / stepNum.toDouble().pow(stepPow)

            sumGrad = sumGrad.minus(gradients[i])
            gradients[i] = calcGrad(createFuncForPoint(row, ans), weights)
            val diff = sumGrad.mult( gradRed ).plus( gradients[i] ).mult( gradStepSize ).div( table.rowsCnt().toDouble() )
            sumGrad = sumGrad.plus(gradients[i])

            weights = weights.minus( diff )
            stepNum += 1

            lastEst = festimation
            val err = calcErr(row, ans, weights)
            festimation = err * lam + (1 - lam) * lastEst
            if (festimation + 100 < bestRes) {
                bestRes = festimation
            }
        }

        myWeights = weights
    }

    override fun predict(table: DataFrame): DataFrame {
        val res = DataFrame(0, 1)

        table.getRows().forEach {row ->
            var ans = 0.0
            ans += myWeights!![0]

            (1 until myWeights!!.size).forEach {i ->
                ans += myWeights!![i] * row[i - 1]
            }
            res.appendRow(listOf(ans))
        }
        return res
    }

    private fun initWeights(table: DataFrame): List<Double> {
        val res = mutableListOf<Double>()
        for (i in 0 until table.colsCnt() + 1) {
            res.add(0.0)
        }
        return res
    }

    private fun calcGrad(f: (List<Double>) -> Double, point: List<Double>): List<Double> {
        val res = mutableListOf<Double>()
        for (i in 0 until point.size) {
            res.add(calcDerivative({x ->
                    val fixedPoint = point.toMutableList()
                    fixedPoint[i] = x
                    f(fixedPoint)
            }, point[i]))
        }
        return res.normalize()
    }

    private fun calcDerivative(f: (Double) -> Double, point: Double): Double {
        var delt = 0.1
        var lastDiff = 1e9
        var lastVal: Double? = null
        val fval = f(point)
        while (lastDiff > derConvEps) {
            val curVal = (f(point + delt) - fval) / delt
            if (lastVal != null) {
                lastDiff = abs(curVal - lastVal)
                delt /= 2
            }
            lastVal = curVal
        }
        return lastVal!!
    }

    private fun createFuncForPoint(row: List<Double>, ans: Double): (List<Double>) -> Double {
        return {weights -> calcErr(row, ans, weights)}
    }

    private fun calcErr(row: List<Double>, ans: Double, weights: List<Double>): Double {
        if (weights.size != row.size + 1) {
            throw IllegalStateException("Wrong sizez")
        }

        var predictedAnswer = 0.0
        for (i in weights.indices) {
            predictedAnswer += if (i == 0) {
                weights[i]
            } else {
                row[i - 1] * weights[i]
            }
        }

        return (predictedAnswer - ans).pow(2)
    }

    private fun List<Double>.mult(v: Double): List<Double> {
        return this.map {it * v}
    }

    private fun List<Double>.plus(other: List<Double>): List<Double> {
        return other.zip(this).map {(a, b) -> a + b}
    }

    private fun List<Double>.div(v: Double): List<Double> {
        return this.map {it / v}
    }

    private fun List<Double>.minus(other: List<Double>): List<Double> {
        return this.zip(other).map {(a, b) -> a - b}
    }

    private fun List<Double>.normalize(): List<Double> {
        val sum2 = this.sumByDouble { it.pow(2) }.pow(0.5)
        return this.div(sum2)
    }

    private fun List<Double>.norm2(): Double {
        return this.sumByDouble { it.pow(2) }
    }
}