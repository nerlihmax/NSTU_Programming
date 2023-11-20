package ru.kheynov.santa.api.v1.routing.utils

import io.ktor.http.*
import io.ktor.server.application.*
import io.ktor.server.auth.*
import io.ktor.server.auth.jwt.*
import io.ktor.server.response.*

fun ApplicationCall.getUserId(): String? =
    principal<JWTPrincipal>()?.payload?.getClaim("userId")?.asString()

fun ApplicationCall.getId(): String? = request.queryParameters["id"]