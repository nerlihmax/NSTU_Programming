package ru.kheynov.hotel.api.routing.utils

import io.ktor.server.application.ApplicationCall
import io.ktor.server.auth.jwt.JWTPrincipal
import io.ktor.server.auth.principal

fun ApplicationCall.getUserId(): String? =
    principal<JWTPrincipal>()?.payload?.getClaim("userId")?.asString()

fun ApplicationCall.getId(): String? = request.queryParameters["id"]