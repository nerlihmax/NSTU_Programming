import kotlinx.serialization.Serializable
import spark.kotlin.get

fun main() {
    get("/objects") {
        response.type("application/json")
        ObjectsContainer.getObjects()
    }
}

@Serializable
data class GraphicalObject(val type: String)

sealed interface Object {
    data class Star()
}

object ObjectsContainer {
    private val objects = mutableListOf<GraphicalObject>()

    fun getObjects(): List<GraphicalObject> = objects
}