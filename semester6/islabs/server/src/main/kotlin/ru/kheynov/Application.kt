package ru.kheynov

import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.callloging.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.plugins.defaultheaders.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import kotlinx.serialization.Serializable
import org.slf4j.event.Level
import java.sql.Connection
import java.sql.DriverManager

fun main() {
    embeddedServer(Netty, port = 8080, host = "0.0.0.0", module = Application::module).start(wait = true)
}

fun Application.module() {
    install(ContentNegotiation) {
        json()
    }
    install(CallLogging) {
        level = Level.INFO
        filter { call -> call.request.path().startsWith("/") }
    }
    install(DefaultHeaders) {
        header("X-Engine", "Ktor") // will send this header with each response
    }
    mainApplication()
}

fun Application.mainApplication() {
    val dbConnection: Connection = connectToPostgres()
//    val cityService = CityService(dbConnection)
    routing {
        get("/") {
            call.respondText("Hello World!")
        }

        // Create city
        post("/cities") {
//            val city = call.receive<City>()
//            val id = cityService.create(city)
//            call.respond(HttpStatusCode.Created, id)
        }
        // Read city
        get("/cities/{id}") {
//            val id = call.parameters["id"]?.toInt() ?: throw IllegalArgumentException("Invalid ID")
            try {
//                val city = cityService.read(id)
//                call.respond(HttpStatusCode.OK, city)
            } catch (e: Exception) {
                call.respond(HttpStatusCode.NotFound)
            }
        }
        // Update city
        put("/cities/{id}") {
//            val id = call.parameters["id"]?.toInt() ?: throw IllegalArgumentException("Invalid ID")
//            val user = call.receive<City>()
//            cityService.update(id, user)
//            call.respond(HttpStatusCode.OK)
        }
        // Delete city
        delete("/cities/{id}") {
//            val id = call.parameters["id"]?.toInt() ?: throw IllegalArgumentException("Invalid ID")
//            cityService.delete(id)
//            call.respond(HttpStatusCode.OK)
        }
    }
}

fun Application.connectToPostgres(): Connection {
    Class.forName("org.postgresql.Driver")
    val url = environment.config.property("postgres.url").getString()
    val user = environment.config.property("postgres.user").getString()
    val password = environment.config.property("postgres.password").getString()

    return DriverManager.getConnection(url, user, password)
}

@Serializable
data class City(val name: String, val population: Int)
class CityService(private val connection: Connection) {
    companion object {
//        private const val CREATE_TABLE_CITIES =
//            "CREATE TABLE CITIES (ID SERIAL PRIMARY KEY, NAME VARCHAR(255), POPULATION INT);"
//        private const val SELECT_CITY_BY_ID = "SELECT name, population FROM cities WHERE id = ?"
//        private const val INSERT_CITY = "INSERT INTO cities (name, population) VALUES (?, ?)"
//        private const val UPDATE_CITY = "UPDATE cities SET name = ?, population = ? WHERE id = ?"
//        private const val DELETE_CITY = "DELETE FROM cities WHERE id = ?"
    }

//    init {
//        val statement = connection.createStatement()
//        statement.executeUpdate(CREATE_TABLE_CITIES)
//    }

    // Create new city
//    suspend fun create(city: City): Int = withContext(Dispatchers.IO) {
//        val statement = connection.prepareStatement(INSERT_CITY, Statement.RETURN_GENERATED_KEYS)
//        statement.setString(1, city.name)
//        statement.setInt(2, city.population)
//        statement.executeUpdate()
//
//        val generatedKeys = statement.generatedKeys
//        if (generatedKeys.next()) {
//            return@withContext generatedKeys.getInt(1)
//        } else {
//            throw Exception("Unable to retrieve the id of the newly inserted city")
//        }
//    }

    // Read a city
//    suspend fun read(id: Int): City = withContext(Dispatchers.IO) {
//        val statement = connection.prepareStatement(SELECT_CITY_BY_ID)
//        statement.setInt(1, id)
//        val resultSet = statement.executeQuery()
//
//        if (resultSet.next()) {
//            val name = resultSet.getString("name")
//            val population = resultSet.getInt("population")
//            return@withContext City(name, population)
//        } else {
//            throw Exception("Record not found")
//        }
//    }

    // Update a city
//    suspend fun update(id: Int, city: City) = withContext(Dispatchers.IO) {
//        val statement = connection.prepareStatement(UPDATE_CITY)
//        statement.setString(1, city.name)
//        statement.setInt(2, city.population)
//        statement.setInt(3, id)
//        statement.executeUpdate()
//    }

    // Delete a city
//    suspend fun delete(id: Int) = withContext(Dispatchers.IO) {
//        val statement = connection.prepareStatement(DELETE_CITY)
//        statement.setInt(1, id)
//        statement.executeUpdate()
//    }
}