import com.github.doyaaaaaken.kotlincsv.dsl.csvWriter
import java.io.File

fun main() {
    val MAX_DEPTH = 5
    val MIN_LEAFS = 3
    val alldata = DataLoader("comp1_train.csv").read()
    val (target, data) = alldata.splitColsBy(1)

    val model = DecisionTreeRegressor(MAX_DEPTH, MIN_LEAFS)

    model.fit(data, target)

    val unknown = DataLoader("comp1_test.csv").read()

    val res = model.predict(unknown)

    println(meanSqError(res, target))

    DataLoader("res.csv").writeCsvIndexed(res, listOf("target", "Id"))
}