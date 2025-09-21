<?php

namespace App\Models\Providers;

class LoginProvider extends BaseApiProvider
{
    public function __construct()
    {
        parent::__construct(config('services.auth.base_url'));
    }

    public function login(array $data)
    {
        return $this->post('/api/auth/login', $data, true);
    }

    public function checkAuth($token)
    {
        return $this->get('/api/auth/me',[
            "Authorization" => $token
        ]);
    }
}
