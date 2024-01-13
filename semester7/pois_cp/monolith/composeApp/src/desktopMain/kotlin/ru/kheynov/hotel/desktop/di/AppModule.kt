package ru.kheynov.hotel.desktop.di

import org.koin.dsl.module
import retrofit2.Retrofit
import ru.kheynov.hotel.shared.data.api.ReservationsAPI
import ru.kheynov.hotel.shared.data.api.UserAPI
import ru.kheynov.hotel.shared.data.repository.ClientReservationsRepository
import ru.kheynov.hotel.shared.data.repository.ClientUsersRepository

const val BASE_URL = "https://hotel.kheynov.ru/api/"

val appModule = module {
    single<UserAPI> {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .build()
            .create(UserAPI::class.java)
    }

    single<ReservationsAPI> {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .build()
            .create(ReservationsAPI::class.java)
    }

    single {
        ClientUsersRepository(get())
    }

    single {
        ClientReservationsRepository(get())
    }
}