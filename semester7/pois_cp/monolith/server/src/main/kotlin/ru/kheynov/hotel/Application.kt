package ru.kheynov.hotel

import io.ktor.server.application.Application
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty
import org.koin.ktor.plugin.Koin
import org.koin.logger.slf4jLogger
import ru.kheynov.hotel.di.appModule
import ru.kheynov.hotel.plugins.configureHTTP
import ru.kheynov.hotel.plugins.configureMonitoring
import ru.kheynov.hotel.plugins.configureRouting
import ru.kheynov.hotel.plugins.configureSecurity
import ru.kheynov.hotel.plugins.configureSerialization
import ru.kheynov.hotel.plugins.configureSockets

fun main() {
    embeddedServer(
        Netty,
        port = System.getenv("SERVER_PORT").toInt(),
        host = "0.0.0.0",
        module = Application::module
    )
        .start(wait = true)
}

fun Application.module() {
    configureDI()
    configureSecurity()
    configureHTTP()
    configureMonitoring()
    configureSerialization()
    configureSockets()
    configureRouting()
}

private fun Application.configureDI() {
    install(Koin) {
        slf4jLogger()
        modules(appModule)
    }
}
