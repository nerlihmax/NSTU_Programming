package ru.kheynov.hotel.api.routing

import io.ktor.server.routing.Route
import io.ktor.server.routing.route
import org.koin.ktor.ext.inject
import ru.kheynov.hotel.domain.useCases.UseCases

fun Route.apiRoutes() {
    route("/api") {
        val useCases by inject<UseCases>()

        configureUserRoutes(useCases)
    }
}