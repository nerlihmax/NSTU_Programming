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
import java.io.File
import java.sql.Connection
import java.sql.DriverManager
import java.time.Instant
import java.util.*
import java.util.concurrent.TimeUnit

fun main() {
    embeddedServer(Netty, port = 8080, host = "0.0.0.0", module = Application::module).start(wait = true) // запускаем сервер
}

@Serializable
data class Client(val host: String, val port: String, val user: String, val password: String, val database: String) // модель клиента

data class ClientSession(val uuid: UUID = UUID.randomUUID()) // сессия клиента

fun Application.module() {
    install(ContentNegotiation) {
        json() // устанавливаем json как формат ответа
    }
    install(Sessions) {
        header<ClientSession>("user_session") // устанавливаем сессию клиента
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
    mainApplication() // запускаем основное приложение
}

fun Application.mainApplication() {
    var client: Client? = null
    var session: ClientSession? = null // сессия клиента
    var dbConnection: Connection? = null // соединение с базой данных
    var dbService: DbService? = null // сервис для работы с базой данных
    routing {
        route("/api") {
            post("/connect") { // подключение к базе данных
                if (client != null && session?.uuid != call.sessions.get<ClientSession>()?.uuid) {
                    call.respond(HttpStatusCode.Forbidden, "Service already in use") // если сервис уже используется, то возвращаем ошибку
                    return@post
                }
                client = call.receiveNullable<Client>() ?: run {
                    call.respond(HttpStatusCode.BadRequest) // если не удалось получить данные клиента, то возвращаем ошибку
                    return@post
                }
                try {
                    dbConnection = connectToPostgres(client!!) //   подключаемся к базе данных
                    dbService = DbService(dbConnection!!, client!!) // создаем сервис для работы с базой данных
                    session = ClientSession() // создаем сессию клиента
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

            post("/disconnect") { // отключение от базы данных
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
            post("/query") { // выполнение запроса
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
                    if (res is DbService.Result.Successful)
                        call.respond(
                            HttpStatusCode.OK,
                            res.data
                        )
                    else { // если запрос не удалось выполнить, то возвращаем ошибку
                        call.respond(HttpStatusCode.BadRequest, "Failed to execute query")
                        return@post
                    }
                } catch (e: Exception) {
                    call.respond(HttpStatusCode.BadRequest, "Failed to execute query")
                    println(e)
                    return@post
                }
            }

            get("/databases") { // получение списка баз данных
                val currentSession = call.sessions.get<ClientSession>()
                println("current session: ${currentSession}, saved session: $session")
                if (session?.uuid != currentSession?.uuid) {
                    call.respond(HttpStatusCode.Forbidden)
                    return@get
                }
                try {
                    val res = dbService!!.fetchDatabases()
                    if (res is DbService.Result.List) call.respond(HttpStatusCode.OK, res.data)
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

            get("/backups") { // получение списка бэкапов
                val currentSession = call.sessions.get<ClientSession>()
                println("current session: ${currentSession}, saved session: $session")
                if (session?.uuid != currentSession?.uuid) {
                    call.respond(HttpStatusCode.Forbidden)
                    return@get
                }
                try {
                    val res = dbService!!.listBackups()
                    call.respond(HttpStatusCode.OK, res)
                    return@get
                } catch (e: Exception) {
                    call.respond(HttpStatusCode.BadRequest, "Failed to fetch backups")
                    println(e)
                    return@get
                }
            }
            post("save") { // создание бэкапа
                val currentSession = call.sessions.get<ClientSession>()
                println("current session: ${currentSession}, saved session: $session")
                if (session?.uuid != currentSession?.uuid) {
                    call.respond(HttpStatusCode.Forbidden)
                    return@post
                }
                try {
                    val res = dbService!!.createBackup()
                    if (res is DbService.Result.Successful)
                        call.respond(HttpStatusCode.OK, "OK")
                    return@post
                } catch (e: Exception) {
                    call.respond(HttpStatusCode.BadRequest, "Failed to dump database")
                    println(e)
                    return@post
                }
            }

            post("restore") { // восстановление бэкапа
                val currentSession = call.sessions.get<ClientSession>()
                println("current session: ${currentSession}, saved session: $session")
                if (session?.uuid != currentSession?.uuid) {
                    call.respond(HttpStatusCode.Forbidden)
                    return@post
                }
                val backupName = call.receiveText()
                try {
                    val res = dbService!!.restoreBackup(backupName)
                    if (res is DbService.Result.Successful)
                        call.respond(HttpStatusCode.OK, "OK")
                    return@post
                } catch (e: Exception) {
                    call.respond(HttpStatusCode.BadRequest, "Failed to restore backup")
                    println(e)
                    return@post
                }
            }
        }
    }
}

fun connectToPostgres(client: Client): Connection { // подключение к базе данных
    Class.forName("org.postgresql.Driver")
    val url = "jdbc:postgresql://${client.host}:${client.port}/${client.database}" // формирование строки подключения
    val user = client.user
    val password = client.password
    return DriverManager.getConnection(url, user, password) // подключение к базе данных
}

class DbService(
    private val connection: Connection,
    private val client: Client,
) {
    suspend fun executeQuery(query: String): Result = withContext(Dispatchers.IO) {
        try {
            val statement = connection.prepareStatement(query) // создание запроса
            if (query.lowercase().run { contains("update") || contains("insert") || contains("delete") }) {
                statement.executeUpdate()
            } else {
                statement.executeQuery()
            }
            val result = statement.resultSet ?: return@withContext Result.Successful(emptyMap()) // получение результата
            val metadata = result.metaData // получение метаданных
            val columns = mutableMapOf<Int, Map<String, String>>()
            var count = 0
            while (result.next()) {
                val row = mutableMapOf<String, String>()
                (1..metadata.columnCount).forEach { idx -> // формирование результата
                    val element: Any? = result.getObject(idx)
                    row[metadata.getColumnName(idx)] = element.toString() // добавление элемента в строку
                }
                columns[count] = row
                count++
            }
            return@withContext Result.Successful(columns.toMap())
        } catch (e: Exception) {
            e.printStackTrace()
            return@withContext Result.Failed(e.localizedMessage)
        }
    }

    sealed interface Result { // результат выполнения запроса
        data class Successful(val data: Map<Int, Map<String, String>>) : Result
        data class List(val data: kotlin.collections.List<String>) : Result
        data class Failed(val reason: String? = null) : Result
    }


    private fun Iterable<String>.runCommands(workingDir: File, envs: Map<String, String>) { // выполнение команд
        this.forEach {
            ProcessBuilder(it.split(" "))
                .directory(workingDir)
                .redirectOutput(ProcessBuilder.Redirect.INHERIT)
                .redirectError(ProcessBuilder.Redirect.INHERIT)
                .apply {
                    environment().putAll(envs)
                }
                .start()
                .waitFor(60, TimeUnit.MINUTES)
        }
    }

    suspend fun fetchDatabases(): Result = withContext(Dispatchers.IO) { // получение списка баз данных
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
            return@withContext Result.Failed(e.localizedMessage)
        }
    }

    suspend fun createBackup(): Result = try { // создание бэкапа
        val backupPath = "/tmp/backups/${client.database}-${Instant.now().epochSecond}.sql" // путь к бэкапу
        withContext(Dispatchers.IO) {
            listOf(
                "/opt/homebrew/bin/pg_dump -h ${client.host} -p ${client.port} -U ${client.user} -f $backupPath -d ${client.database}",
            ).runCommands(File("/tmp/backups"), mapOf("PGPASSWORD" to client.password))// выполнение команд
        }
        Result.Successful(emptyMap())
    } catch (e: Exception) {
        e.printStackTrace()
        Result.Failed(e.localizedMessage)
    }

    fun listBackups(): List<String> = // получение списка бэкапов
        File("/tmp/backups").walkTopDown().filter { it.name.endsWith(".sql") && it.name.startsWith(client.database) }
            .map { it.name }.toList()

    suspend fun restoreBackup(backupName: String): Result = try { // восстановление бэкапа
        withContext(Dispatchers.IO) {
            executeQuery("drop schema if exists public cascade; create schema public;") // удаление схемы
            listOf( // выполнение команд
                "/opt/homebrew/bin/psql -h ${client.host} -p ${client.port} -U ${client.user} -d ${client.database} -f /tmp/backups/$backupName"
            ).runCommands(File("/tmp/backups"), mapOf("PGPASSWORD" to client.password))
        }
        Result.Successful(emptyMap())
    } catch (e: Exception) {
        e.printStackTrace()
        Result.Failed(e.localizedMessage)
    }

}