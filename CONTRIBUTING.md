# Contribuição

Preserve `main` e `develop`: sem push direto, force push ou exclusão. Cada alteração deve usar uma nova branch e um pull request com descrição e evidências de validação.

```bash
git fetch origin
git switch -c codex/nome-da-alteracao origin/develop
# Implemente e verifique.
python scripts/validate_data.py
python -m unittest discover -s tests -v
git add <arquivos-alterados>
git commit -m "Descreve a alteração"
git push -u origin codex/nome-da-alteracao
```

Abra um pull request para `develop`; revise o diff, os testes e o impacto nos dados antes da integração. Promova versões estáveis por pull request de `develop` para `main`.

Não edite manualmente o CSV em `data/raw/`. Transformações futuras devem escrever em `data/processed/`. Documente decisões de limpeza e mudanças intencionais de versão. Não versione credenciais, ambientes virtuais ou modelos gerados nesta etapa.

Estas são convenções do projeto. Proteções de branches no GitHub dependem da configuração administrativa do repositório e não são instaladas por estes arquivos.
