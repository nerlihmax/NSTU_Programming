package ru.kheynov

import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.callloging.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.plugins.cors.routing.*
import io.ktor.server.plugins.defaultheaders.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import io.ktor.server.sessions.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import org.slf4j.event.Level
import java.sql.Connection
import java.sql.DriverManager
import java.util.*

fun main() {
    embeddedServer(Netty, port = 8080, host = "0.0.0.0", module = Application::module).start(wait = true)
}

@Serializable
data class Client(val host: String, val port: String, val user: String, val password: String, val database: String)

data class ClientSession(val uuid: UUID = UUID.randomUUID())

fun Application.module() {
    install(ContentNegotiation) {
        json()
    }
    install(Sessions) {
        header<ClientSession>("user_session")
    }
    install(CallLogging) {
        level = Level.INFO
        filter { call -> call.request.path().startsWith("/") }
    }
    install(DefaultHeaders) {
        header("X-Engine", "Ktor") // will send this header with each response
    }
    install(CORS) {
        anyHost()
        HttpMethod.DefaultMethods.forEach(::allowMethod)
        allowHeader(HttpHeaders.Authorization)
        allowHeader(HttpHeaders.ContentType)
        allowHeader("user_session")
        exposeHeader("user_session")
    }
    mainApplication()
}

fun Application.mainApplication() {
    var client: Client? = null
    var session: ClientSession? = null
    var dbConnection: Connection? = null
    var dbService: DbService? = null
    routing {
        get("/") {
            call.respondText("Hello World!")
        }
        route("/api") {
            post("/connect") {
                if (client != null && session?.uuid != call.sessions.get<ClientSession>()?.uuid) {
                    call.respond(HttpStatusCode.Forbidden, "Service already in use")
                    return@post
                }
                client = call.receiveNullable<Client>() ?: run {
                    call.respond(HttpStatusCode.BadRequest)
                    return@post
                }
                try {
                    dbConnection = connectToPostgres(client!!)
                    dbService = DbService(dbConnection!!)
                    session = ClientSession()
                } catch (e: Exception) {
                    dbConnection = null
                    dbService = null
                    session = null
                    client = null
                    call.respond(HttpStatusCode.NotAcceptable, "Failed to connect")
                    println(e)
                    return@post
                }
                call.sessions.set(session)
                call.respond(HttpStatusCode.OK)
            }

            post("/disconnect") {
                val currentSession = call.sessions.get<ClientSession>()
                println("current session: ${currentSession}, saved session: $session")
                if (session?.uuid != currentSession?.uuid) {
                    call.respond(HttpStatusCode.Forbidden)
                    return@post
                }
                session = null
                dbConnection = null
                dbService = null
                client = null
                call.sessions.clear<ClientSession>()
                call.respond(HttpStatusCode.OK)
            }
            post("/query") {
                val currentSession = call.sessions.get<ClientSession>()
                println("current session: ${currentSession}, saved session: $session")
                if (session?.uuid != currentSession?.uuid) {
                    call.respond(HttpStatusCode.Forbidden)
                    return@post
                }
                if (client == null || dbConnection == null || dbService == null) {
                    call.respond(HttpStatusCode.Unauthorized)
                    return@post
                }
                val query = call.receiveText()
                println(query)
                try {
                    val res = dbService!!.executeQuery(query)
                    if (res is DbService.Companion.Result.Successful) call.respond(HttpStatusCode.OK, res.data)
                    else {
                        call.respond(HttpStatusCode.BadRequest, "Failed to execute query")
                        return@post
                    }
                } catch (e: Exception) {
                    call.respond(HttpStatusCode.BadRequest, "Failed to execute query")
                    println(e)
                    return@post
                }
            }

            get("/databases") {
                val currentSession = call.sessions.get<ClientSession>()
                println("current session: ${currentSession}, saved session: $session")
                if (session?.uuid != currentSession?.uuid) {
                    call.respond(HttpStatusCode.Forbidden)
                    return@get
                }
                try {
                    val res = dbService!!.fetchDatabases()
                    if (res is DbService.Companion.Result.List) call.respond(HttpStatusCode.OK, res.data)
                    else {
                        call.respond(HttpStatusCode.BadRequest, "Failed to execute query")
                        return@get
                    }
                } catch (e: Exception) {
                    call.respond(HttpStatusCode.BadRequest, "Failed to execute query")
                    println(e)
                    return@get
                }
            }
        }
    }
}

fun connectToPostgres(client: Client): Connection {
    Class.forName("org.postgresql.Driver")
    val url = "jdbc:postgresql://${client.host}:${client.port}/${client.database}"
    val user = client.user
    val password = client.password
    return DriverManager.getConnection(url, user, password)
}

class DbService(private val connection: Connection) {
    companion object {
        sealed interface Result {
            data class Successful(val data: Map<Int, Map<String, String>>) : Result
            data class List(val data: kotlin.collections.List<String>) : Result
            object Failed : Result
        }
    }

    suspend fun executeQuery(query: String): Result = withContext(Dispatchers.IO) {
        try {
            val statement = connection.prepareStatement(query)
            statement.executeQuery()
            val result = statement.resultSet
            val metadata = result.metaData
            val columns = mutableMapOf<Int, Map<String, String>>()
            var count = 0
            while (result.next()) {
                val row = mutableMapOf<String, String>()
                (1..metadata.columnCount).forEach { idx ->
                    val element: Any? = result.getObject(idx)
                    row[metadata.getColumnName(idx)] = element.toString()
                }
                columns[count] = row
                count++
            }
            return@withContext Result.Successful(columns.toMap())
        } catch (e: Exception) {
            e.printStackTrace()
            return@withContext Result.Failed
        }
    }

    suspend fun fetchDatabases(): Result = withContext(Dispatchers.IO) {
        try {
            val query = "SELECT datname From pg_database WHERE pg_database.datistemplate=false;"
            val statement = connection.prepareStatement(query)
            statement.executeQuery()
            val result = statement.resultSet
            val names = mutableListOf<String>()
            while (result.next()) {
                names.add(result.getString("datname"))
            }
            return@withContext Result.List(names.toList())
        } catch (e: Exception) {
            e.printStackTrace()
            return@withContext Result.Failed
        }
    }
}