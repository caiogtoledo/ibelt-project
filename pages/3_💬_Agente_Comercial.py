import utils
import os
import faiss
import numpy as np
import streamlit as st
from openai import OpenAI


class CommercialAgentChatbot:
    def __init__(self):
        utils.configure_openai()
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        # self.index, self.embeddings = self.criar_indice_faiss(faq_data)
        self.historico_conversa = [
            {"role": "assistant", "content": "Olá! Como posso te ajudar hoje?"}]
        self.lead_data = ""

    def responder_pergunta_com_historico(self, pergunta):
        prompt = "\n".join(
            [f"Usuário: {pergunta}\n", f"<LEAD_DATA>\n{self.lead_data}\n</LEAD_DATA>", "Se a informação já está no LEAD_DATA, não pergunte novamente"])
        system_prompt = """
        Você é um chatbot Agente Comercial da Pieracciani, especializado em identificar potenciais clientes que se encaixam no perfil ideal para nossa equipe comercial. 
        Seu objetivo é interagir de forma amigável e eficiente com visitantes do site, 
        coletando informações relevantes para determinar se eles devem ser encaminhados para um atendimento humano mais aprofundado.
        Ao conversar com os usuários, você deve:
        #Coletar Informações:
        - Pergunte sobre o nome e a empresa do usuário.
        - Identifique o setor de atuação e o tamanho da empresa.
        - Entenda as necessidades ou desafios que o usuário está enfrentando.
        - É uma empresa industrial, comercial ou de seviços?
        - A empresa está no regime fiscal de lucro real?
        - A empresa possui técnicos trabalhando em projetos?
        - Os projetos são de produtos novos ou parcialmente modificados?
        - Projetos de processo?
        - As atividades de desenvolvimento ou preparação das mudanças de produto e processo são feitas internamente?
        - A empresa tem parceiros para ajudar nesses desenvolvimentos?
        - Quantas pessoas na empresa trabalham totalmente ou parcialmente para essas mudanças?
        - Uma quantidade aproximada desses técnicos poderia ser estimada?
        - Quantos projetos de inovações ou mudanças técnicas aproximadamente estão sendo realizados?
        - Nos passe uma ideia do tipo de trabalho de renovação de produto ou serviço que são efetuados.
        #Qualificar o Lead:
        Avalie se o setor, tamanho da empresa e necessidades do usuário estão alinhados com o perfil de cliente ideal da Pieracciani.
        Utilize perguntas adicionais para clarificar qualquer dúvida sobre o potencial do lead.
        #Encaminhar ou Agradecer:
        Se o usuário se qualificar como um potencial cliente, informe que você irá encaminhar suas informações para a equipe comercial, que entrará em contato em breve.
        Caso contrário, agradeça o interesse e forneça informações úteis ou sugestões para futuras interações.
        
        #Regras:
        - Não forneça informações pessoais ou confidenciais.
        - Não responda sobre preços, valores e investimentos nos nossos serviços, isso é um especialista que irá responder.
        - Não pergunte para o usuário uma informação que já foi coletada e está entre as tags <LEAD_DATA> e </LEAD_DATA>.
        - Você não explica nada, apenas pergunta sobre o que o usuário está falando.
        """
        try:
            resposta = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                ]+self.historico_conversa+[{"role": "user", "content": prompt}],
            )
            resposta_texto = resposta.choices[0].message.content.strip()
            self.historico_conversa.append(
                {"role": "user", "content": pergunta})
            self.historico_conversa.append(
                {"role": "system", "content": resposta_texto})
            return resposta_texto
        except Exception as e:
            return f"Erro ao obter resposta: {str(e)}"

    def setLeadData(self, lead_data):
        self.lead_data = lead_data


# Configuração da interface do Streamlit
st.set_page_config(page_title="Agente Comercial", page_icon="📄")
st.image("https://dev.pierxinovacao.com.br/assets/img/logo.svg", width=120)
st.header('Fale com o Agente Comercial')


# Instancia o chatbot
chatbot = CommercialAgentChatbot()


def save_lead_data(history_lead_data, new_lead_data):
    print(f"Acessando lead_data da classe: {history_lead_data}")
    prompt = "\n".join(
        [f"Informações do Lead atual: {history_lead_data}\n", f"Nova resposta do Usuário: {new_lead_data}\n",
            "Não apague informações já existentes, apenas concatene as novas informações em um novo JSON."])
    system_prompt = """
    Você é um assistente que armazena informações sobre leads qualificados para a equipe comercial da Pieracciani.
    Você irá receber as informações já existentes e as novas.
    Sua função é concatenar a informação de forma estruturada no formato JSON.
    #Regras:
    - Não sobrescreva informações já existentes.
    - Use APENAS informações relevantes para a equipe comercial.
    - Sempre use o formato JSON como resposta
    """
    try:
        resposta = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")).chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
            ]+[{"role": "user", "content": prompt}],
        )
        resposta_texto = resposta.choices[0].message.content.strip()
        resposta_texto = resposta_texto.replace(
            "```json", "").replace("```", "").strip()
        return resposta_texto
    except Exception as e:
        return f"Erro ao obter resposta: {str(e)}"

# Função principal para o aplicativo


def main():
    # Inicializa uma lista para armazenar o histórico de mensagens
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append(
        ("assistant", "Olá, sou o Agente Comercial da Pieracciani. \nQual o seu nome?"))

    user_query = st.chat_input(
        placeholder="Como podemos te auxiliar?")

    if user_query:
        # Adiciona a mensagem do usuário ao histórico
        st.session_state.chat_history.append(("user", user_query))

        # Obtém a resposta do chatbot
        resposta = chatbot.responder_pergunta_com_historico(user_query)

        # Coleta informações do lead
        print(
            f"Lead data antes de passar pro save_lead_data: {chatbot.lead_data}")
        new_lead_data = save_lead_data(chatbot.lead_data, user_query)
        chatbot.setLeadData(new_lead_data)
        print("Lead Data Dentro da Main: ", chatbot.lead_data)
        # Adiciona a resposta do assistente ao histórico
        st.session_state.chat_history.append(("assistant", resposta))

        if st.session_state.chat_history[0] == st.session_state.chat_history[1]:
            st.session_state.chat_history.pop(0)

    # Exibe todas as mensagens no histórico
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)


if __name__ == "__main__":
    main()
