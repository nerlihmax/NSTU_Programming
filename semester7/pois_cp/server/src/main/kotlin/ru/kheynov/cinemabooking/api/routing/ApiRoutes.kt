package ru.kheynov.cinemabooking.api.routing

import io.ktor.server.routing.*
import org.koin.ktor.ext.inject
import ru.kheynov.cinemabooking.domain.repositories.UsersRepository
import ru.kheynov.cinemabooking.domain.useCases.UseCases

fun Route.apiRoutes() {
    route("/api") {
        val useCases by inject<UseCases>()

        configureUserRoutes(useCases)
    }
}