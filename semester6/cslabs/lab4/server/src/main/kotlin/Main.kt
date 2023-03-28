import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.eclipse.jetty.http.HttpStatus
import spark.Spark.path
import spark.kotlin.delete
import spark.kotlin.get
import spark.kotlin.post
import java.io.File

fun main() {
    ObjectsContainer.restoreFromFile()
    get("/objects") {
        response.type("application/json")
        Json.encodeToString(ObjectsContainer.getObjects())
    }
    path("/object") {
        get("/get") {
            val id = queryParams("id").run { if (isBlank()) return@get "Id is empty" else toInt() }
            response.type("application/json")
            Json.run {
                encodeToString(ObjectsContainer.getObject(id) ?: kotlin.run {
                        response.type("plain/text")
                        response.status(HttpStatus.NOT_FOUND_404)
                        return@get ""
                    })
            }
        }
        post("/add") {
            try {
                val obj = Json.decodeFromString<GraphicalObject>(request.body())
                if (ObjectsContainer.addObject(obj)) {
                    response.status(HttpStatus.OK_200)
                    return@post ""
                }
                response.status(HttpStatus.CONFLICT_409)
            } catch (e: Exception) {
                e.printStackTrace()
                response.status(HttpStatus.BAD_REQUEST_400)
            }
            return@post ""
        }
        delete("/remove") {
            val id = queryParams("id").run { if (isBlank()) return@delete "Id is empty" else toInt() }
            if (!ObjectsContainer.deleteObject(id)) {
                response.status(HttpStatus.NOT_FOUND_404)
            }
            return@delete ""
        }
    }

    Runtime.getRuntime().addShutdownHook(Thread {
        ObjectsContainer.saveToFile()
    })
}

@Serializable
sealed interface GraphicalObject {
    @Serializable
    @SerialName("star")
    data class Star(
        val x: Int,
        val y: Int,
        val width: Int,
        val height: Int,
        val r: Int,
        val g: Int,
        val b: Int,
        val numberOfVertices: Int,
    ) : GraphicalObject

    @Serializable
    @SerialName("smiley")
    data class Smiley(
        val x: Int,
        val y: Int,
        val width: Int,
        val height: Int,
        val r: Int,
        val g: Int,
        val b: Int,
        val vx: Int,
        val vy: Int,
    ) : GraphicalObject
}

object ObjectsContainer {
    private val objects = mutableListOf<GraphicalObject>()
    fun getObjects(): List<GraphicalObject> = objects.toList()
    fun addObject(obj: GraphicalObject) = objects.add(obj)
    fun getObject(idx: Int) = objects.getOrNull(idx)
    fun deleteObject(idx: Int): Boolean {
        return try {
            objects.removeAt(idx)
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    fun saveToFile() {
        File("./resources/backup.json").printWriter().use { out ->
            Json.encodeToString(objects).let { out.write(it) }
        }
    }

    fun restoreFromFile() {
        File("./resources/backup.json").bufferedReader().use { reader ->
            try {
                Json.decodeFromString<List<GraphicalObject>>(reader.readText()).let { objects.addAll(it) }
            } catch (e: Exception) {
                println("Nothing to restore")
            }
        }
    }
}