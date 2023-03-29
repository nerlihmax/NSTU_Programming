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

fun main() { // точка входа в программу
    ObjectsContainer.restoreFromFile() // восстановление объектов из файла

    get("/objects") {
        response.type("application/json")
        Json.encodeToString(ObjectsContainer.getObjects()) // Получение всех объектов
    }
    path("/object") {
        get("/get") {
            val id =
                queryParams("id").run { if (isBlank()) return@get "Id is empty" else toInt() }
            // получение id объекта из параметров запроса и проверка на пустоту

            response.type("application/json")
            Json.run {
                encodeToString(ObjectsContainer.getObject(id) // получение объекта по id
                    ?: kotlin.run { // если объекта с таким id нет, то возвращается пустая строка и код 404
                        response.type("plain/text")
                        response.status(HttpStatus.NOT_FOUND_404)
                        return@get ""
                    })
            }
        }
        post("/add") {
            try {
                val obj = Json.decodeFromString<GraphicalObject>(request.body()) // десериализация объекта из тела запроса
                if (ObjectsContainer.addObject(obj)) { // добавление объекта в список
                    response.status(HttpStatus.OK_200)
                    return@post ""
                }
                response.status(HttpStatus.CONFLICT_409) // если объект добавить не удалось, то возвращается код 409
            } catch (e: Exception) {
                e.printStackTrace()
                response.status(HttpStatus.BAD_REQUEST_400) // если объект не удалось десериализовать, то возвращается код 400
            }
            return@post ""
        }
        delete("/remove") {
            val id = queryParams("id").run { if (isBlank()) return@delete "Id is empty" else toInt() }
            if (!ObjectsContainer.deleteObject(id)) { // удаление объекта по id
                response.status(HttpStatus.NOT_FOUND_404) // если объекта с таким id нет, то возвращается код 404
            }
            return@delete ""
        }
    }

    Runtime.getRuntime().addShutdownHook(Thread { // сохранение объектов в файл при завершении программы
        ObjectsContainer.saveToFile()
    })
}

@Serializable // аннотация для сериализации в json
sealed interface GraphicalObject { // интерфейс графического объекта (data class аналог record в Java)
    @Serializable
    @SerialName("star")
    data class Star( // звезда
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
    data class Smiley( // смайлик
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

object ObjectsContainer { // object -- синглтон (аналог static в Java)
    private val objects = mutableListOf<GraphicalObject>() // список объектов

    fun getObjects(): List<GraphicalObject> = objects.toList() // получение списка объектов
    fun addObject(obj: GraphicalObject) = objects.add(obj) // добавление объекта
    fun getObject(idx: Int) = objects.getOrNull(idx) // получение объекта по индексу
    fun deleteObject(idx: Int): Boolean { // удаление объекта по индексу
        return try {
            objects.removeAt(idx)
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    fun saveToFile() { // сохранение объектов в файл
        File("./resources/backup.json").printWriter().use { out -> // открытие файла
            Json.encodeToString(objects).let { out.write(it) } // сериализация списка объектов в json и запись в файл
        }
    }

    fun restoreFromFile() { // восстановление объектов из файла
        File("./resources/backup.json").bufferedReader().use { reader -> // открытие файла
            try {
                Json.decodeFromString<List<GraphicalObject>>(reader.readText()).let { objects.addAll(it) }
            // десериализация списка объектов из json и добавление в список
            } catch (e: Exception) {
                println("Nothing to restore") // если файл пустой, то ничего не делаем
            }
        }
    }
}