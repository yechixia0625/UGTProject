%% Key generation
function [publicKey, privateKey] = paillierKeyGen(bitLength)
    % Generates two large prime numbers p and q
    p = generateLargePrime(bitLength);
    q = generateLargePrime(bitLength);
    % Calculate n and λ
    n = p * q;
    lambda = lcm(p - 1, q - 1);
    % Generates g
    g = n + 1;
    % The private key is λ
    privateKey = lambda;
    % The public key is (n, g)
    publicKey = struct('n', n, 'g', g);
end

% Generate large prime numbers
function prime = generateLargePrime(bitLength)
    prime = 2;
    while ~isprime(prime)
        prime = randi([2^(bitLength-1), 2^bitLength]);
    end
end

%% Encryption
function encryptedMessage = paillierEncrypt(message, publicKey)
    % Generates a random number r, r < n
    r = randi([1, publicKey.n-1]);
    % Calculate ciphertext c
    c = mod(publicKey.g^message * r^publicKey.n, publicKey.n^2);
    encryptedMessage = c;
end

%% Decryption
function decryptedMessage = paillierDecrypt(encryptedMessage, privateKey, publicKey)
    % Calculate L(c^λ mod n^2)
    c_lambda = mod(encryptedMessage^privateKey, publicKey.n^2);
    L_c_lambda = (c_lambda - 1) / publicKey.n;
    % Calculate L(g^λ mod n^2)
    g_lambda = mod(publicKey.g^privateKey, publicKey.n^2);
    L_g_lambda = (g_lambda - 1) / publicKey.n;
    % Decryption Message
    decryptedMessage = mod(L_c_lambda * modInverse(L_g_lambda, publicKey.n), publicKey.n);
end

% Inverse mod
function inv = modInverse(a, n)
    [g, x, ~] = gcd(a, n);
    if g == 1
        inv = mod(x, n);
    else
        error('No modular inverse exists'); 
    end
end
