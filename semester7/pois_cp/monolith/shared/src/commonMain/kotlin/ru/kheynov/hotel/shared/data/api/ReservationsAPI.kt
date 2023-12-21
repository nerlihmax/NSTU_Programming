package ru.kheynov.hotel.shared.data.api

import retrofit2.http.Body
import retrofit2.http.DELETE
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Query
import ru.kheynov.hotel.shared.data.models.reservations.ReserveRoomRequest
import ru.kheynov.hotel.shared.domain.entities.Hotel
import ru.kheynov.hotel.shared.domain.entities.RoomInfo
import ru.kheynov.hotel.shared.domain.entities.RoomReservationInfo
import java.time.LocalDate

interface ReservationsAPI {
    @GET("reservations/list")
    suspend fun getReservations(
        @Query("userId") userId: String? = null,
    ): List<RoomReservationInfo>

    @POST("reservations")
    suspend fun addReservation(
        @Body reservation: ReserveRoomRequest,
    ): String

    @DELETE("reservations")
    suspend fun deleteReservation(
        @Query("id") id: String,
    ): String

    @GET("hotels")
    suspend fun getHotels(): List<Hotel>

    @GET("rooms")
    suspend fun getRooms(
        @Query("hotel") hotelId: Int,
    ): List<RoomInfo>

    @GET("rooms/occupancy")
    suspend fun getRoomsOccupancy(
        @Query("room") roomId: String,
    ): List<ClosedRange<LocalDate>>
}