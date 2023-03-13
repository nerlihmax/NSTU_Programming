sudo apt update -y 
sudo apt upgrade -y
sudo apt install zsh -y
sudo chsh /bin/zsh
sudo apt install git curl -y
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:neovim-ppa/stable -y
sudo apt update
sudo apt install neovim -y
sh -c 'curl -fLo "${XDG_DATA_HOME:-$HOME/.local/share}"/nvim/site/autoload/plug.vim --create-dirs \
       https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'

curl -L https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh | sh
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k

echo '
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="powerlevel10k/powerlevel10k"
plugins=(
	git
	sudo
	zsh-autosuggestions
	zsh-syntax-highlighting
)

alias zshconfig="nvim ~/.zshrc"
alias src="source ~/.zshrc"

alias vim="nvim"

source $ZSH/oh-my-zsh.sh
[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh
' > ~/.zshrc

source ~/.zshrc

mkdir ~/.config/nvim

echo ':set number
:set autoindent
:set tabstop=4
:set shiftwidth=4
:set smarttab
:set softtabstop=4
:set mouse=a
:set relativenumber
:set nohlsearch
:set shell=/usr/bin/zsh
:set noswapfile
:set fileformat=unix
:set encoding=utf-8
:set clipboard=unnamedplus

call plug#begin()

Plug 'vim-airline/vim-airline'
Plug 'preservim/nerdtree'
Plug 'tpope/vim-surround'
Plug 'tpope/vim-commentary'
Plug 'rafi/awesome-vim-colorschemes'
Plug 'https://github.com/ryanoasis/vim-devicons'

Plug 'tpope/vim-commentary'
Plug 'tc50cal/vim-terminal'
Plug 'jiangmiao/auto-pairs'
Plug 'joshdick/onedark.vim'
Plug 'neovim/nvim-lspconfig'

Plug 'kyazdani42/nvim-web-devicons' " icons

Plug 'octol/vim-cpp-enhanced-highlight'
let g:cpp_class_scope_highlight = 1
let g:cpp_class_decl_highlight  = 1

let g:airline_theme='onedark'

call plug#end()

nnoremap <C-a> ggVG<CR>
xnoremap p pgddvy

nnoremap <C-t> :NvimTreeToggle<CR>
inoremap <silent><expr> <TAB> pumvisible() ? coc#_select_confirm() : "\<C-g>u\<TAB>"

:set termguicolors
colorscheme onedark' > ~/.config/nvim/init.vim

sudo apt install nginx

sudo apt install apt-transport-https ca-certificates curl software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"

sudo apt update

sudo apt-cache policy docker-ce

sudo apt install docker-ce -y

sudo curl -L "https://github.com/docker/compose/releases/download/1.26.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose

sudo usermod -aG docker ${USER}

sudo su - ${USER}

id -nG
