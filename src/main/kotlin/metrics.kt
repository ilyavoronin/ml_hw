import java.lang.IllegalStateException
import kotlin.math.pow

fun meanSqError(data1: DataFrame, data2: DataFrame): Double {
    if (data1.rowsCnt() != data2.rowsCnt() || data1.colsCnt() != 1 || data2.colsCnt() != 1) {
        throw IllegalStateException("Incompatible dimensions")
    }

    val l1 = data1.tr()[0]
    val l2 = data2.tr()[0]
    var res = 0.0
    l1.zip(l2).forEach{(a, b) ->
        res += (a - b).pow(2)
    }
    return res / data1.rowsCnt()
}

fun Collection<Double>.mean(): Double {
    return this.sum() / this.size
}

fun aucRoc(expected: DataFrame, actual: DataFrame): Double {

    val exp = expected.tr()[0]
    val act = actual.tr()[0]

    val zipped = exp.zip(act).toMutableList()

    zipped.shuffle()
    zipped.sortBy {it.second}

    val k1 = zipped.count {it.first == 1.0}
    val k0 = zipped.size - k1

    var res = 0.0

    var curra = 0.0

    zipped.reversed().forEach {(a, _) ->
        if (a == 1.0) {
            curra += 1.0 / k1
        } else {
            res += curra / k0
        }
    }

    return res
}