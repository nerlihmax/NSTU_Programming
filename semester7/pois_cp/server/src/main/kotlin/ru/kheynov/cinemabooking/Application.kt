package ru.kheynov.cinemabooking

import io.ktor.server.application.*
import org.koin.ktor.plugin.Koin
import org.koin.logger.slf4jLogger
import ru.kheynov.cinemabooking.di.appModule
import ru.kheynov.cinemabooking.plugins.*

fun main(args: Array<String>) {
    io.ktor.server.netty.EngineMain.main(args)
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