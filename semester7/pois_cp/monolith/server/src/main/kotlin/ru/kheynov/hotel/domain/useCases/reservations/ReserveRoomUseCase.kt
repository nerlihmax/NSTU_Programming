package ru.kheynov.hotel.domain.useCases.reservations

import org.koin.core.component.KoinComponent
import org.koin.core.component.inject
import ru.kheynov.hotel.data.entities.Employees.user
import ru.kheynov.hotel.shared.domain.entities.RoomReservation
import ru.kheynov.hotel.shared.domain.repository.ReservationsRepository
import ru.kheynov.hotel.shared.domain.repository.UsersRepository
import java.time.LocalDate
import java.util.UUID

class ReserveRoomUseCase : KoinComponent {
    private val reservationsRepository: ReservationsRepository by inject()
    private val usersRepository: UsersRepository by inject()

    sealed interface Result {
        data class Successful(val reservationId: String) : Result
        data object Failed : Result
        data object UserNotExists : Result
        data object RoomNotExists : Result
        data object RoomNotAvailable : Result
    }

    suspend operator fun invoke(
        userId: String,
        roomId: String,
        from: LocalDate,
        to: LocalDate,
    ): Result {
        val user = usersRepository.getUserByID(userId) ?: return Result.UserNotExists
        val room = reservationsRepository.getRoomByID(roomId) ?: return Result.RoomNotExists

        if (reservationsRepository
                .getRoomOccupancy(roomId)
                .any { it.contains(from) || it.contains(to) }
        ) return Result.RoomNotAvailable

        val reservationId = UUID.randomUUID().toString().substring(0, 6)

        val reservation = RoomReservation(
            id = reservationId,
            room = room,
            user = user,
            from = from,
            to = to,
        )

        return if (reservationsRepository.reserveRoom(reservation)) {
            Result.Successful(reservationId)
        } else Result.Failed
    }
}