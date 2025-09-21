import {BaseApi} from "./BaseApi.js";

export class AuthApi extends BaseApi{
    constructor() {
        super('https://lct-2025')
    }

    async login(data) {
        try {
            super.httpMethod = 'post';
            super.sourceUrl = '/api/login';
            super.data = data;
            let token = await super.createRequest();
            localStorage.setItem('bearerToken', token.data.access_token);
        } catch (e) {
            return e;
        }
    }

    async checkAuth() {
        try {
            super.httpMethod = 'get';
            super.sourceUrl = '/api/checkAuth';
            let token = localStorage.getItem('bearerToken');
            super.headers = {'Authorization': 'Bearer ' + token};
            return await super.createRequest();
        } catch (e) {
            return e;
        }
    }

}
