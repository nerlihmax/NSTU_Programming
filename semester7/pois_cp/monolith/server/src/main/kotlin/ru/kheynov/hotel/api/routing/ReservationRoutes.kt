package ru.kheynov.hotel.api.routing

import io.ktor.http.HttpStatusCode
import io.ktor.server.application.call
import io.ktor.server.auth.authenticate
import io.ktor.server.request.receiveNullable
import io.ktor.server.response.respond
import io.ktor.server.routing.Route
import io.ktor.server.routing.delete
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.route
import ru.kheynov.hotel.api.routing.utils.getUserId
import ru.kheynov.hotel.domain.useCases.UseCases
import ru.kheynov.hotel.domain.useCases.reservations.DeleteReservationUseCase
import ru.kheynov.hotel.domain.useCases.reservations.GetHotelsUseCase
import ru.kheynov.hotel.domain.useCases.reservations.GetRoomsReservationsUseCase
import ru.kheynov.hotel.domain.useCases.reservations.GetRoomsUseCase
import ru.kheynov.hotel.domain.useCases.reservations.GetUsersReservationsUseCase
import ru.kheynov.hotel.domain.useCases.reservations.ReserveRoomUseCase
import ru.kheynov.hotel.shared.data.models.reservations.ReserveRoomRequest
import java.time.LocalDate

fun Route.configureReservationsRoutes(
    useCases: UseCases,
) {
    configureReservationRoutes(useCases)
    configureHotelRoutes(useCases)
    configureRoomRoutes(useCases)
}

fun Route.configureReservationRoutes(
    useCases: UseCases,
) {
    route("/reservations") {
        authenticate {
            get("/list") {
                val selfId = call.getUserId() ?: run {
                    call.respond(HttpStatusCode.Unauthorized, "No access token provided")
                    return@get
                }
                val userId = call.request.queryParameters["userId"]

                when (val res = useCases.getUsersReservationsUseCase(selfId, userId)) {
                    GetUsersReservationsUseCase.Result.Empty -> {
                        call.respond(HttpStatusCode.NoContent, "Empty")
                        return@get
                    }

                    is GetUsersReservationsUseCase.Result.Successful -> {
                        call.respond(HttpStatusCode.OK, res.reservations)
                        return@get
                    }

                    GetUsersReservationsUseCase.Result.UserNotExists -> {
                        call.respond(HttpStatusCode.BadRequest, "User not exists")
                        return@get
                    }

                    GetUsersReservationsUseCase.Result.Forbidden -> {
                        call.respond(HttpStatusCode.Forbidden, "Forbidden")
                        return@get
                    }
                }
            }
        }

        authenticate {
            post {
                val userId = call.getUserId() ?: run {
                    call.respond(HttpStatusCode.Unauthorized, "No access token provided")
                    return@post
                }
                val reservation = call.receiveNullable<ReserveRoomRequest>() ?: run {
                    call.respond(HttpStatusCode.BadRequest)
                    return@post
                }
                when (val res = useCases.reserveRoomUseCase(
                    userId,
                    reservation.roomId,
                    LocalDate.parse(reservation.from),
                    LocalDate.parse(reservation.to),
                )) {
                    ReserveRoomUseCase.Result.Failed -> {
                        call.respond(HttpStatusCode.InternalServerError, "Something went wrong")
                        return@post
                    }

                    ReserveRoomUseCase.Result.RoomNotAvailable -> {
                        call.respond(HttpStatusCode.BadRequest, "Room not available")
                        return@post
                    }

                    ReserveRoomUseCase.Result.RoomNotExists -> {
                        call.respond(HttpStatusCode.BadRequest, "Room not exists")
                        return@post
                    }

                    is ReserveRoomUseCase.Result.Successful -> {
                        call.respond(HttpStatusCode.OK, res.reservationId)
                        return@post
                    }

                    ReserveRoomUseCase.Result.UserNotExists -> {
                        call.respond(HttpStatusCode.BadRequest, "User not exists")
                        return@post
                    }
                }
            }
        }
        authenticate {
            delete {
                val userId = call.getUserId() ?: run {
                    call.respond(HttpStatusCode.Unauthorized, "No access token provided")
                    return@delete
                }
                val reservationId = call.request.queryParameters["id"] ?: run {
                    call.respond(HttpStatusCode.BadRequest, "No reservation id provided")
                    return@delete
                }
                when (val res = useCases.deleteReservationUseCase(userId, reservationId)) {
                    DeleteReservationUseCase.Result.Failed -> {
                        call.respond(HttpStatusCode.InternalServerError, "Something went wrong")
                        return@delete
                    }

                    DeleteReservationUseCase.Result.Forbidden -> {
                        call.respond(HttpStatusCode.Forbidden, "Forbidden")
                        return@delete
                    }

                    DeleteReservationUseCase.Result.ReservationNotExists -> {
                        call.respond(HttpStatusCode.BadRequest, "Reservation not exists")
                        return@delete
                    }

                    is DeleteReservationUseCase.Result.Successful -> {
                        call.respond(HttpStatusCode.OK, res.reservationId)
                        return@delete
                    }

                    DeleteReservationUseCase.Result.UserNotExists -> {
                        call.respond(HttpStatusCode.BadRequest, "User not exists")
                        return@delete
                    }
                }
            }
        }
    }
}

fun Route.configureHotelRoutes(
    useCases: UseCases,
) {
    route("/hotels") {
        authenticate {
            get {
                call.getUserId() ?: run {
                    call.respond(HttpStatusCode.Unauthorized, "No access token provided")
                    return@get
                }
                when (val res = useCases.getHotelsUseCase()) {
                    GetHotelsUseCase.Result.Failed -> {
                        call.respond(HttpStatusCode.InternalServerError, "Something went wrong")
                        return@get
                    }

                    is GetHotelsUseCase.Result.Successful -> {
                        call.respond(HttpStatusCode.OK, res.data)
                        return@get
                    }

                    GetHotelsUseCase.Result.Empty -> {
                        call.respond(HttpStatusCode.NoContent, "Empty")
                        return@get
                    }
                }
            }
        }
    }
}

fun Route.configureRoomRoutes(
    useCases: UseCases,
) {
    route("/rooms") {
        authenticate {
            get {
                call.getUserId() ?: run {
                    call.respond(HttpStatusCode.Unauthorized, "No access token provided")
                    return@get
                }
                val hotelId = call.request.queryParameters["hotel"]?.toInt() ?: run {
                    call.respond(HttpStatusCode.BadRequest, "No hotel id provided")
                    return@get
                }
                when (val res = useCases.getRoomsUseCase(hotelId)) {
                    GetRoomsUseCase.Result.Empty -> {
                        call.respond(HttpStatusCode.NoContent, "Empty")
                        return@get
                    }

                    GetRoomsUseCase.Result.Failed -> {
                        call.respond(HttpStatusCode.InternalServerError, "Something went wrong")
                        return@get
                    }

                    is GetRoomsUseCase.Result.Successful -> {
                        call.respond(HttpStatusCode.OK, res.data)
                        return@get
                    }

                    GetRoomsUseCase.Result.UnknownHotel -> {
                        call.respond(HttpStatusCode.BadRequest, "Unknown hotel")
                        return@get
                    }
                }
            }
        }
        authenticate {
            get("/occupancy") {
                val roomId = call.request.queryParameters["room"] ?: run {
                    call.respond(HttpStatusCode.BadRequest, "No room id provided")
                    return@get
                }
                when (val res = useCases.getRoomsReservationsUseCase(
                    roomId = roomId,
                )) {
                    GetRoomsReservationsUseCase.Result.Empty -> {
                        call.respond(HttpStatusCode.NoContent, "Empty")
                        return@get
                    }

                    GetRoomsReservationsUseCase.Result.Failed -> {
                        call.respond(HttpStatusCode.InternalServerError, "Something went wrong")
                        return@get
                    }

                    is GetRoomsReservationsUseCase.Result.Successful -> {
                        call.respond(HttpStatusCode.OK, res.data)
                        return@get
                    }

                    GetRoomsReservationsUseCase.Result.UnknownRoom -> {
                        call.respond(HttpStatusCode.BadRequest, "Unknown room")
                        return@get
                    }
                }
            }
        }
    }
}