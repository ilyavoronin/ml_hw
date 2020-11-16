package classification

import DataFrame
import MLModel
import Params
import java.lang.IllegalStateException
import java.util.*
import kotlin.math.*

class LogisticRegression(
        var initialStepSize: Double,
        var stepPow: Double,
        private var algConvEps: Double,
        var ta: Double
) : MLModel {
    constructor(): this(1.0, 1.9, 0.01, 1.0)

    private var myRandom = Random()

    private var myWeights: List<Double>? = null

    fun setSeed(seed: Long) {
        myRandom = Random(seed)
    }

    override fun setParams(params: Params) {
        params.getParam("initialStepSize")?.let { initialStepSize = it }
        params.getParam("algConvEps")?.let { algConvEps = it }
        params.getParam("stepPow")?.let { stepPow = it }
    }

    override fun fit(table: DataFrame, target: DataFrame) {
        var weights = initWeights(table)
        var festimation = 1e9
        var lastEst = festimation + 1e9

        var bestRes = 1e9

        var stepNum = 1

        var kstop = 0

        while (kstop < 100) {
            if (abs(lastEst - festimation) < algConvEps) {
                kstop += 1
            } else {
                kstop = 0
            }
            if (stepNum % 1000 == 0) {
                println(festimation)
            }

            val gradStepSize = initialStepSize / stepNum.toDouble().pow(stepPow)


            val sumGrad = calcGrad(table, target, weights)

            val diff = sumGrad.mult( gradStepSize )

            weights = weights.minus( diff )
            stepNum += 1

            lastEst = festimation
            festimation = 0.0
            table.getRows().forEachIndexed {i, row ->
                festimation += calcErr(row, target[i][0], weights)
            }
            festimation += weights.norm2() * ta / 2
            festimation /= table.rowsCnt()
        }

        println(festimation)

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

            ans = 1 / (1 + exp(-ans))
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

    private fun calcGrad(table: DataFrame, target: DataFrame, point: List<Double>): List<Double> {
        val res = mutableListOf<Double>()
        for (i in 0 until point.size) {
            var a = 0.0
            for (j in 0 until table.rowsCnt()) {
                a += (calch(table[j], point) - target[j][0]) * if (i == 0) 1.0 else table[j][i - 1]
            }
            a = a / table.rowsCnt()
            res.add(a)
        }
        return if (res.all {it == 0.0}) {
            res.plus(point.mult(ta))
        } else {
            res.plus(point.mult(ta)).normalize()
        }
    }

    private fun calcErr(row: List<Double>, ans: Double, weights: List<Double>): Double {
        if (weights.size != row.size + 1) {
            throw IllegalStateException("Wrong sizez")
        }

        val h = calch(row, weights);

        val res =  -ans * ln(h) - (1 - ans) * ln(1 - h)
        return res
    }

    private fun calch(row: List<Double>, weights: List<Double>): Double {

        var h = 0.0
        for (i in weights.indices) {
            h += if (i == 0) {
                weights[i]
            } else {
                row[i - 1] * weights[i]
            }
        }

        h = 1 / (1 + exp(h))
        h = min(h, 0.9999)
        h = max(h, 0.0001)
        return h
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