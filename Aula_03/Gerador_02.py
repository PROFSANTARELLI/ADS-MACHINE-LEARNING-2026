import pandas as pd

data = {
    'texto': [
        "Meu produto ainda não CHEGOU!!!", 
        "Gostaria de saber o preço do frete.",
        "ESTOU MUITO BRAVO COM O ATRASO",
        "Como faço para rastrear meu pedido?",
        "O boleto já foi pago, cadê meu acesso?",
        "Quero cancelar essa compra agora!",
        "Qual o horário de atendimento de vocês?",
        "Vocês aceitam Pix como forma de pagamento?"
    ],
    'rotulo': [1, 0, 1, 0, 1, 1, 0, 0] # 1: Reclamação/Urgente, 0: Informação
}

df = pd.DataFrame(data)
df.to_csv('mensagens_suporte.csv', index=False)
print("Dataset 'mensagens_suporte.csv' gerado!")
