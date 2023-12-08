package ru.kheynov.hotel.jwt.token

import ru.kheynov.hotel.shared.jwt.TokenPair

interface TokenService {
    fun generateTokenPair(config: TokenConfig, vararg claims: TokenClaim): TokenPair
}