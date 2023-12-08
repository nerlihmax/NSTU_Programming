package ru.kheynov.hotel.domain.useCases.reservations

import org.koin.core.component.KoinComponent
import org.koin.core.component.inject
import ru.kheynov.hotel.shared.domain.entities.RoomReservationInfo
import ru.kheynov.hotel.shared.domain.repository.ReservationsRepository

class GetRoomsUseCase : KoinComponent {
    private val reservationsRepository: ReservationsRepository by inject()

    sealed interface Result {
        data class Successful(val data: List<RoomReservationInfo>) : Result
        data object UnknownHotel : Result
        data object Failed : Result
        data object Empty : Result
    }

    suspend operator fun invoke(hotelId: Int): Result {
        val hotel = reservationsRepository.getHotels().find { it.id == hotelId }
            ?: return Result.UnknownHotel

        val rooms = reservationsRepository.getAvailableRooms(hotel)
        return try {
            if (rooms.isEmpty()) Result.Empty
            else Result.Successful(rooms)
        } catch (e: Exception) {
            Result.Failed
        }
    }
}