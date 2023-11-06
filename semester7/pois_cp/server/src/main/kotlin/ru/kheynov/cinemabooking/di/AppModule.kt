package ru.kheynov.cinemabooking.di

import org.koin.dsl.module
import org.ktorm.database.Database
import ru.kheynov.cinemabooking.jwt.hashing.BcryptHashingService
import ru.kheynov.cinemabooking.jwt.hashing.HashingService
import ru.kheynov.cinemabooking.jwt.token.JwtTokenService
import ru.kheynov.cinemabooking.jwt.token.TokenConfig
import ru.kheynov.cinemabooking.jwt.token.TokenService

val appModule = module {
    single {
        Database.connect(
            url = System.getenv("DATABASE_CONNECTION_STRING"),
            driver = "org.postgresql.Driver",
            user = System.getenv("POSTGRES_NAME"),
            password = System.getenv("POSTGRES_PASSWORD"),
        )
    }

    single {
        TokenConfig(
            issuer = System.getenv("JWT_ISSUER"),
            audience = System.getenv("JWT_AUDIENCE"),
            accessLifetime = System.getenv("JWT_ACCESS_LIFETIME").toLong(),
            refreshLifetime = System.getenv("JWT_REFRESH_LIFETIME").toLong(),
            secret = System.getenv("JWT_SECRET"),
        )
    }

    single<TokenService> { JwtTokenService() }

    single<HashingService> { BcryptHashingService() }

}