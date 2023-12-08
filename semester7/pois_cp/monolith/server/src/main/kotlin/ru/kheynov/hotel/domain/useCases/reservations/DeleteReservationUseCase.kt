package ru.kheynov.hotel.domain.useCases.reservations

import org.koin.core.component.KoinComponent
import org.koin.core.component.inject
import ru.kheynov.hotel.shared.domain.repository.ReservationsRepository
import ru.kheynov.hotel.shared.domain.repository.UsersRepository

class DeleteReservationUseCase : KoinComponent {
    private val usersRepository: UsersRepository by inject()
    private val reservationsRepository: ReservationsRepository by inject()

    sealed interface Result {
        data class Successful(val reservationId: String) : Result
        data object Failed : Result
        data object Forbidden : Result
        data object UserNotExists : Result
        data object ReservationNotExists : Result
    }

    suspend operator fun invoke(
        userId: String,
        reservationId: String,
    ): Result {
        val user = usersRepository.getUserByID(userId) ?: return Result.UserNotExists
        val reservation = reservationsRepository.getReservationByID(reservationId)
            ?: return Result.ReservationNotExists
        if (user != reservation.user) return Result.Forbidden
        return if (reservationsRepository.cancelReservation(reservationId)) {
            Result.Successful(reservationId)
        } else Result.ReservationNotExists
    }
}