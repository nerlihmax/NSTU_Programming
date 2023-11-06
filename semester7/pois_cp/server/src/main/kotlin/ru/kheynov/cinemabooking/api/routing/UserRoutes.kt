package ru.kheynov.cinemabooking.api.routing

import io.ktor.http.*
import io.ktor.server.application.*
import io.ktor.server.auth.*
import io.ktor.server.auth.jwt.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import ru.kheynov.cinemabooking.api.requests.users.UpdateUserRequest
import ru.kheynov.cinemabooking.api.requests.users.auth.LoginViaEmailRequest
import ru.kheynov.cinemabooking.api.requests.users.auth.RefreshTokenRequest
import ru.kheynov.cinemabooking.api.requests.users.auth.SignUpViaEmailRequest
import ru.kheynov.cinemabooking.domain.entities.UserDTO
import ru.kheynov.cinemabooking.domain.useCases.DeleteUserUseCase
import ru.kheynov.cinemabooking.domain.useCases.GetUserDetailsUseCase
import ru.kheynov.cinemabooking.domain.useCases.UpdateUserUseCase
import ru.kheynov.cinemabooking.domain.useCases.UseCases
import ru.kheynov.cinemabooking.domain.useCases.auth.LoginViaEmailUseCase
import ru.kheynov.cinemabooking.domain.useCases.auth.RefreshTokenUseCase
import ru.kheynov.cinemabooking.domain.useCases.auth.SignUpViaEmailUseCase

fun Route.configureUserRoutes(
    useCases: UseCases,
) {
    configureAuthRoutes(useCases)

    route("/user") {

        authenticate {
            get {
                val userId = call.principal<JWTPrincipal>()?.payload?.getClaim("userId")?.asString() ?: run {
                    call.respond(HttpStatusCode.Unauthorized, "No access token provided")
                    return@get
                }
                when (val res = useCases.getUserDetailsUseCase(userId)) {
                    GetUserDetailsUseCase.Result.Failed -> {
                        call.respond(HttpStatusCode.InternalServerError, "Something went wrong")
                        return@get
                    }

                    is GetUserDetailsUseCase.Result.Successful -> {
                        call.respond(HttpStatusCode.OK, res.user)
                        return@get
                    }

                    GetUserDetailsUseCase.Result.UserNotFound -> {
                        call.respond(HttpStatusCode.BadRequest, "User not found")
                        return@get
                    }
                }
            }
        }
        authenticate {
            delete {
                val userId = call.principal<JWTPrincipal>()?.payload?.getClaim("userId")?.asString() ?: run {
                    call.respond(HttpStatusCode.Unauthorized, "No access token provided")
                    return@delete
                }

                when (useCases.deleteUserUseCase(userId)) {
                    DeleteUserUseCase.Result.Failed -> {
                        call.respond(HttpStatusCode.InternalServerError, "Something went wrong")
                        return@delete
                    }

                    DeleteUserUseCase.Result.Successful -> {
                        call.respond(HttpStatusCode.OK)
                        return@delete
                    }

                    DeleteUserUseCase.Result.UserNotExists -> {
                        call.respond(HttpStatusCode.BadRequest, "User not found")
                        return@delete
                    }
                }
            }
        }

        authenticate {
            patch {
                val userId = call.principal<JWTPrincipal>()?.payload?.getClaim("userId")?.asString() ?: run {
                    call.respond(HttpStatusCode.Unauthorized, "No access token provided")
                    return@patch
                }

                val userUpdate = call.receiveNullable<UpdateUserRequest>()?.let {
                    UserDTO.UpdateUser(
                        username = it.username,
                    )
                } ?: run {
                    call.respond(HttpStatusCode.BadRequest)
                    return@patch
                }
                if (userUpdate.username == null) {
                    call.respond(HttpStatusCode.BadRequest, "No data provided")
                    return@patch
                }

                when (useCases.updateUserUseCase(userId, userUpdate)) {
                    UpdateUserUseCase.Result.Failed -> {
                        call.respond(HttpStatusCode.InternalServerError, "Something went wrong")
                        return@patch
                    }

                    UpdateUserUseCase.Result.Successful -> {
                        call.respond(HttpStatusCode.OK)
                        return@patch
                    }

                    UpdateUserUseCase.Result.UserNotExists -> {
                        call.respond(HttpStatusCode.BadRequest, "User not found")
                        return@patch
                    }

                    UpdateUserUseCase.Result.AvatarNotFound -> {
                        call.respond(HttpStatusCode.BadRequest, "Avatar not found")
                        return@patch
                    }
                }
            }
        }
    }
}

private fun Route.configureAuthRoutes(useCases: UseCases) {
    route("/auth") {
        post("/email/register") { // Register user
            val clientId = call.request.headers["client-id"] ?: run {
                call.respond(HttpStatusCode.BadRequest, "No client id provided")
                return@post
            }

            val registerRequest = call.receiveNullable<SignUpViaEmailRequest>()?.let {
                UserDTO.UserEmailSignUp(
                    username = it.username,
                    password = it.password,
                    email = it.email,
                    clientId = clientId,
                )
            } ?: run {
                call.respond(HttpStatusCode.BadRequest)
                return@post
            }

            when (val res = useCases.signUpViaEmailUseCase(registerRequest)) {
                SignUpViaEmailUseCase.Result.Failed -> {
                    call.respond(HttpStatusCode.InternalServerError)
                    return@post
                }

                is SignUpViaEmailUseCase.Result.Successful -> {
                    call.respond(HttpStatusCode.OK, res.tokenPair)
                    return@post
                }

                SignUpViaEmailUseCase.Result.UserAlreadyExists -> {
                    call.respond(HttpStatusCode.Conflict, "User already exists")
                    return@post
                }
            }
        }

        post("/email/login") {
            val clientId = call.request.headers["client-id"] ?: run {
                call.respond(HttpStatusCode.BadRequest, "No client id provided")
                return@post
            }
            val (email, password) = call.receiveNullable<LoginViaEmailRequest>() ?: run {
                call.respond(HttpStatusCode.BadRequest)
                return@post
            }

            when (
                val res = useCases.loginViaEmailUseCase(
                    email = email,
                    password = password,
                    clientId = clientId,
                )
            ) {
                LoginViaEmailUseCase.Result.Failed -> {
                    call.respond(HttpStatusCode.InternalServerError)
                    return@post
                }

                is LoginViaEmailUseCase.Result.Success -> {
                    call.respond(HttpStatusCode.OK, res.tokenPair)
                    return@post
                }

                LoginViaEmailUseCase.Result.Forbidden -> {
                    call.respond(HttpStatusCode.Forbidden, "Wrong password or email")
                    return@post
                }
            }
        }

        post("/refresh") {
            val oldRefreshToken = call.receiveNullable<RefreshTokenRequest>()?.oldRefreshToken ?: run {
                call.respond(HttpStatusCode.BadRequest, "No refresh token provided")
                return@post
            }

            when (
                val res = useCases.refreshTokenUseCase(
                    oldRefreshToken = oldRefreshToken,
                )
            ) {
                is RefreshTokenUseCase.Result.Success -> {
                    call.respond(HttpStatusCode.OK, res.tokenPair)
                    return@post
                }

                RefreshTokenUseCase.Result.NoRefreshTokenFound -> {
                    call.respond(HttpStatusCode.BadRequest, "Invalid refresh token")
                    return@post
                }

                RefreshTokenUseCase.Result.RefreshTokenExpired -> {
                    call.respond(HttpStatusCode.BadRequest, "Refresh token expired")
                    return@post
                }

                RefreshTokenUseCase.Result.Forbidden -> {
                    call.respond(HttpStatusCode.Forbidden, "Forbidden")
                    return@post
                }

                RefreshTokenUseCase.Result.Failed -> {
                    call.respond(HttpStatusCode.InternalServerError)
                    return@post
                }
            }
        }
    }
}