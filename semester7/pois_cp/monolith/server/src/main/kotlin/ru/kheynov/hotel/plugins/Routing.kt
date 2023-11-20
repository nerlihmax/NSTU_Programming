package ru.kheynov.cinemabooking.plugins

import io.ktor.server.application.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import ru.kheynov.cinemabooking.api.routing.apiRoutes

fun Application.configureRouting() {
    routing {
        get("/") {
            call.respondText("Hello World!")
        }
        //configuring api routes
        apiRoutes()
    }
}
