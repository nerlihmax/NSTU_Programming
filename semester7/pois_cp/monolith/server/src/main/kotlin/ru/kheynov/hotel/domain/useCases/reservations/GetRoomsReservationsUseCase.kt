package ru.kheynov.hotel.domain.useCases.reservations

import org.koin.core.component.KoinComponent
import org.koin.core.component.inject
import ru.kheynov.hotel.shared.domain.repository.ReservationsRepository
import ru.kheynov.hotel.shared.domain.repository.UsersRepository
import java.time.LocalDate

class GetRoomsReservationsUseCase : KoinComponent {
    private val reservationsRepository: ReservationsRepository by inject()

    sealed interface Result {
        data class Successful(val data: List<ClosedRange<LocalDate>>) : Result
        data object UnknownRoom : Result
        data object Failed : Result
        data object Empty : Result
    }

    suspend operator fun invoke(roomId: String): Result {
        reservationsRepository.getRoomByID(roomId)
            ?: return Result.UnknownRoom

        val rooms = reservationsRepository.getRoomOccupancy(roomId)
        return try {
            if (rooms.isEmpty()) Result.Empty
            else Result.Successful(rooms)
        } catch (e: Exception) {
            Result.Failed
        }
    }
}