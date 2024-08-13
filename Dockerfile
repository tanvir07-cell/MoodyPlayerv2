FROM node:20-alpine as builder

WORKDIR /src

COPY --chown=node:node package*.json ./

RUN npm ci 

COPY --chown=node:node . .

RUN npm run build


FROM nginx:alpine

COPY --from=builder /src/dist /usr/share/nginx/html