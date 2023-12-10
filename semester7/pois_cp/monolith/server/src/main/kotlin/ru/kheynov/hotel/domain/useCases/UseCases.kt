package ru.kheynov.hotel.domain.useCases

import ru.kheynov.hotel.domain.useCases.auth.LoginViaEmailUseCase
import ru.kheynov.hotel.domain.useCases.auth.RefreshTokenUseCase
import ru.kheynov.hotel.domain.useCases.auth.SignUpViaEmailUseCase
import ru.kheynov.hotel.domain.useCases.reservations.DeleteReservationUseCase
import ru.kheynov.hotel.domain.useCases.reservations.GetHotelsUseCase
import ru.kheynov.hotel.domain.useCases.reservations.GetRoomsReservationsUseCase
import ru.kheynov.hotel.domain.useCases.reservations.GetRoomsUseCase
import ru.kheynov.hotel.domain.useCases.reservations.GetUsersReservationsUseCase
import ru.kheynov.hotel.domain.useCases.reservations.ReserveRoomUseCase
import ru.kheynov.hotel.domain.useCases.users.DeleteUserUseCase
import ru.kheynov.hotel.domain.useCases.users.GetUserDetailsUseCase
import ru.kheynov.hotel.domain.useCases.users.UpdateUserUseCase

class UseCases {
    val signUpViaEmailUseCase = SignUpViaEmailUseCase()
    val loginViaEmailUseCase = LoginViaEmailUseCase()
    val refreshTokenUseCase = RefreshTokenUseCase()
    val deleteUserUseCase = DeleteUserUseCase()
    val updateUserUseCase = UpdateUserUseCase()
    val getUserDetailsUseCase = GetUserDetailsUseCase()
    val reserveRoomUseCase = ReserveRoomUseCase()
    val getHotelsUseCase = GetHotelsUseCase()
    val getRoomsUseCase = GetRoomsUseCase()
    val deleteReservationUseCase = DeleteReservationUseCase()
    val getUsersReservationsUseCase = GetUsersReservationsUseCase()
    val getRoomsReservationsUseCase = GetRoomsReservationsUseCase()
}