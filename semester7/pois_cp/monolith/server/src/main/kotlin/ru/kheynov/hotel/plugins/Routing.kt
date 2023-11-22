package ru.kheynov.hotel.plugins

import io.ktor.server.application.Application
import io.ktor.server.application.call
import io.ktor.server.response.respondText
import io.ktor.server.routing.get
import io.ktor.server.routing.routing
import ru.kheynov.hotel.api.routing.apiRoutes

fun Application.configureRouting() {
    routing {
        get("/") {
            call.respondText("Hello World!")
        }
        //configuring api routes
        apiRoutes()
    }
}
