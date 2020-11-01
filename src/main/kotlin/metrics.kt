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