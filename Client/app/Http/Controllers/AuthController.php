<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Providers\LoginProvider;

class AuthController extends Controller
{
    private loginProvider $loginProvider;
    public function __construct()
    {
        $this->loginProvider = new LoginProvider();
    }

    public function login(Request $request)
    {
        $data = [
            'username' => $request->input('login'),
            'password' => $request->input('password')
        ];
        return $this->loginProvider->login($data);
    }

    public function checkAuth(Request $request)
    {
        return $this->loginProvider->checkAuth($request->header('Authorization'));
    }
}
