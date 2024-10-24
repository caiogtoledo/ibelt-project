import utils
import os
import faiss
import numpy as np
import streamlit as st
from openai import OpenAI


class CustomDataChatbot:
    def __init__(self, faq_data):
        self.faq_data = faq_data
        utils.configure_openai()
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.index, self.embeddings = self.criar_indice_faiss(faq_data)
        self.historico_conversa = []

    def criar_indice_faiss(self, faq_data, arquivo_indice='indice_faiss.index'):
        # Verifica se o arquivo do índice já existe
        if os.path.exists(arquivo_indice):
            # Carrega o índice FAISS do arquivo
            index = faiss.read_index(arquivo_indice)
            print("Índice FAISS carregado do arquivo.")
        else:
            # Calcula os embeddings, pois o arquivo não existe
            embeddings = np.array([self.obter_embedding_real(pergunta)
                                   for pergunta, _ in faq_data]).astype('float32')
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)

            # Salva o índice FAISS no arquivo
            faiss.write_index(index, arquivo_indice)
            print("Índice FAISS criado e salvo no arquivo.")

        return index, embeddings if 'embeddings' in locals() else None

    def obter_embedding_real(self, text):
        model = "text-embedding-3-small"
        text = text.replace("\n", " ")
        embedding = self.client.embeddings.create(
            input=[text], model=model).data[0].embedding
        return embedding

    def encontrar_resposta(self, pergunta):
        pergunta_embedding = self.obter_embedding_real(pergunta)
        _, indices = self.index.search(np.array([pergunta_embedding]), 1)
        return self.faq_data[indices[0][0]][1]

    def responder_pergunta_com_historico(self, pergunta):
        resposta_relevante = self.encontrar_resposta(pergunta)
        prompt = "\n".join([f"Usuário: {pergunta}\n", f"Conteúdo relevante para responder o usuário: {resposta_relevante}\n",
                           "Reponda a pergunta com base no conteúdo relevante, mas pode complementar a resposta para deixar mais completa"])
        system_prompt = """
        Você é um assistente útil. Responda a pergunta de acordo com o último conteúdo da Resposta relevante.
        Avalie o conteúdo relevante, caso não ache coerente com a última pergunta, diga 'Não tenho informações suficientes para responder essa pergunta.'
        Responda apenas a última pergunta feita pelo usuário, ou seja, caso não faça sentido continuar o raciocínio da conversa, foque no último assunto abordado pelo usuário.
        Pode complementar a resposta com base no conteúdo relevante e adicionar algo a mais para deixar mais completo.
        Se comporte como um chatbot, irei enviar a pergunta do usuário e um conteúdo relevante para se basear, responda o que um atendente responderia.
        Nunca diga o porquê você não sabe responder a pergunta, apenas responda com base no conteúdo relevante ou a frase 'Não tenho informações suficientes para responder essa pergunta.' e nada mais.
        """
        try:
            resposta = self.client.chat.completions.create(
                model="gpt-4o",
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


# Configuração da interface do Streamlit
st.set_page_config(page_title="PierX AI", page_icon="📄")
st.image("https://dev.pierxinovacao.com.br/assets/img/logo.svg", width=120)
st.header('Fale com o Consultor PierX AI')
st.write('Pergunte o que quiser sobre a PierX')

# Exemplo de perguntas e respostas fictícias
faq_data = [
    ("Como posso redefinir minha senha?",
     "Você pode redefinir sua senha clicando em 'Esqueci minha senha' na página de login."),
    ("Quais são os métodos de pagamento aceitos?",
     "Aceitamos cartões de crédito, PayPal e transferências bancárias."),
    ("Como posso entrar em contato com o suporte?",
     "Você pode entrar em contato com o suporte via e-mail ou chat ao vivo em nosso site."),
    ("Existe um aplicativo móvel disponível?",
     "Sim, nosso aplicativo móvel está disponível para iOS e Android."),
    ("Como faço para cancelar minha assinatura?",
     "Para cancelar sua assinatura, vá até 'Configurações' e clique em 'Cancelar assinatura'."),
    ("Quais são os benefícios de usar a plataforma?",
     "A plataforma oferece suporte especializado, simplificação de processos e maximização de incentivos fiscais para inovação."),
    ("Como a plataforma garante a segurança dos meus dados?",
     "Utilizamos criptografia avançada e seguimos rigorosos padrões de segurança para proteger seus dados."),
    ("A plataforma é compatível com quais navegadores?",
     "Nossa plataforma é compatível com os navegadores mais populares, incluindo Chrome, Firefox, Safari e Edge."),
    ("É possível integrar a plataforma com outros softwares?",
     "Sim, oferecemos integrações com diversos softwares de contabilidade e gestão de projetos."),
    ("Há um período de teste gratuito disponível?",
     "Sim, oferecemos um período de teste gratuito de 14 dias para novos usuários."),
    ("Quais tipos de projetos são elegíveis para incentivos fiscais?",
     "Projetos que envolvem pesquisa, desenvolvimento e inovação tecnológica são geralmente elegíveis."),
    ("A plataforma oferece suporte para empresas de todos os tamanhos?",
     "Sim, nossa plataforma é projetada para atender empresas de todos os tamanhos, desde startups até grandes corporações."),
    ("Como posso atualizar minhas informações de pagamento?",
     "Você pode atualizar suas informações de pagamento na seção 'Faturamento' do seu perfil."),
    ("Existe algum treinamento disponível para novos usuários?",
     "Sim, oferecemos webinars e tutoriais para ajudar novos usuários a se familiarizarem com a plataforma."),
    ("Como posso acompanhar o status do meu pedido de incentivo?",
     "Você pode acompanhar o status do seu pedido na seção 'Meus Pedidos' do seu painel de controle."),
    ("A plataforma está disponível em quais idiomas?",
     "Atualmente, nossa plataforma está disponível em português, inglês e espanhol."),
    ("Como posso adicionar novos membros da equipe à minha conta?",
     "Você pode adicionar novos membros da equipe na seção 'Equipe' do seu perfil."),
    ("Qual é o custo da assinatura mensal?",
     "Os preços variam de acordo com o plano escolhido. Consulte nossa página de preços para mais detalhes."),
    ("A plataforma oferece suporte em tempo real?",
     "Sim, oferecemos suporte em tempo real via chat durante o horário comercial."),
    ("Como posso enviar feedback sobre a plataforma?",
     "Você pode enviar seu feedback através do formulário de contato disponível em nosso site."),
    ("A plataforma ajuda a identificar oportunidades de incentivos fiscais?",
     "Sim, nossa plataforma possui ferramentas que ajudam a identificar e maximizar oportunidades de incentivos fiscais."),
    ("Quais são os requisitos para se qualificar para incentivos fiscais?",
     "Os requisitos podem variar, mas geralmente incluem a realização de atividades de P&D e inovação."),
    ("Como posso acessar relatórios de desempenho dos meus projetos?",
     "Relatórios de desempenho estão disponíveis na seção 'Relatórios' do seu painel de controle."),
    ("Existe algum custo adicional além da assinatura?",
     "Não, todos os custos estão incluídos na assinatura, a menos que você opte por serviços adicionais."),
    ("A plataforma oferece suporte para preenchimento de formulários de incentivo?",
     "Sim, nossa equipe pode ajudar no preenchimento e submissão de formulários de incentivo."),
    ("Como posso alterar meu plano de assinatura?",
     "Você pode alterar seu plano de assinatura na seção 'Plano' do seu perfil."),
    ("Os dados inseridos na plataforma são compartilhados com terceiros?",
     "Não, seus dados são confidenciais e não são compartilhados com terceiros sem seu consentimento."),
    ("A plataforma oferece alguma garantia de sucesso na obtenção de incentivos?",
     "Embora ofereçamos suporte e ferramentas para maximizar suas chances, não podemos garantir o sucesso devido a fatores externos."),
    ("Como posso participar de webinars e eventos da plataforma?",
     "Você pode se inscrever em webinars e eventos através da seção 'Eventos' do nosso site."),
    ("A plataforma oferece alguma certificação para usuários?",
     "Sim, oferecemos certificações após a conclusão de determinados treinamentos e cursos oferecidos pela plataforma.")
]

# Instancia o chatbot
chatbot = CustomDataChatbot(faq_data)

# Função principal para o aplicativo


def main():
    # Inicializa uma lista para armazenar o histórico de mensagens
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.chat_input(
        placeholder="Tem alguma dúvida sobre a PierX?")

    if user_query:
        # Adiciona a mensagem do usuário ao histórico
        st.session_state.chat_history.append(("user", user_query))

        # Obtém a resposta do chatbot
        resposta = chatbot.responder_pergunta_com_historico(user_query)

        # Adiciona a resposta do assistente ao histórico
        st.session_state.chat_history.append(("assistant", resposta))

    # Exibe todas as mensagens no histórico
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)


if __name__ == "__main__":
    main()
