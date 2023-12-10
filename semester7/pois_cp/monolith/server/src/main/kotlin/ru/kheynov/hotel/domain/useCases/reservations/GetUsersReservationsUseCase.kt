package ru.kheynov.hotel.domain.useCases.reservations

import org.koin.core.component.KoinComponent
import org.koin.core.component.inject
import ru.kheynov.hotel.shared.domain.entities.RoomReservationInfo
import ru.kheynov.hotel.shared.domain.repository.ReservationsRepository
import ru.kheynov.hotel.shared.domain.repository.UsersRepository

class GetUsersReservationsUseCase : KoinComponent {
    private val reservationsRepository: ReservationsRepository by inject()
    private val usersRepository: UsersRepository by inject()

    sealed interface Result {
        data class Successful(val reservations: List<RoomReservationInfo>) : Result
        data object Empty : Result
        data object Forbidden : Result
        data object UserNotExists : Result
    }

    suspend operator fun invoke(
        selfId: String,
        userId: String?,
    ): Result {
        if (selfId != userId && !usersRepository.isUserEmployee(selfId)) return Result.Forbidden
        val user = usersRepository.getUserByID(userId ?: selfId) ?: return Result.UserNotExists
        val reservations =
            reservationsRepository.getUsersReservations(user).let { reservationInfos ->
                if (selfId != userId) reservationInfos.map { it.copy(user = null) }
                else reservationInfos
            }
        return if (reservations.isEmpty()) return Result.Empty
        else Result.Successful(reservations)
    }
}